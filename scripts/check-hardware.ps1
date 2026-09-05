#Requires -Version 5.1
<#
    Ayris - прогон тестов, которым нужна настоящая машина.

    С 05.09.2026 маркер `hardware` идёт в обычном `scripts/check.sh` сам:
    микрофон, экран и Диспетчер учётных данных отвечают без спроса, а если
    устройства нет — тест говорит «скип» и почему. Этот скрипт остался для двух
    вещей, которых там нет: доустановить движки речи с моделями и разрешить
    тест с настоящим писком в колонках (AYRIS_TEST_SPEAKERS он ставит сам).

    Зачем отдельный скрипт: десять тестов помечены маркером `hardware` и в CI
    не запускаются, потому что раннеру нечем их выполнить. Девять из десяти
    закрыты в песочнице Claude (движки и модели туда доустановлены), а два
    там ничего не проверяют по существу:

      * микрофон - в песочнице есть ALSA null-устройство, оно отдаёт тишину и
        не соблюдает тайминг, поэтому мерить по нему уровень шума в комнате
        бессмысленно;
      * колонки - того же свойства: проверяется, что звук ОБРЫВАЕТСЯ мгновенно
        по «стоп», а на null-устройстве обрывается что угодно, поэтому там
        этот тест пропускает сам себя.

    Что делает скрипт: доустанавливает в _tools\win-python движки речи и
    скачивает модели (~280 МБ, всё в папку проекта, в систему ничего не
    ставится), затем запускает pytest -m hardware.

    Во время прогона: две секунды тишины для замера шума комнаты и короткий
    тихий гудок в колонках.

    Запуск (одной строкой в PowerShell):
      powershell -ExecutionPolicy Bypass -File "E:\мистер бит ест рис\scripts\check-hardware.ps1"

    Ключи:
      -Only mic         только микрофон (без скачивания моделей, быстро)
      -Only speakers    только колонки
      -Only engines     только движки речи, без вмешательства руками
      -SkipInstall      ничего не доустанавливать, гнать на том, что есть
#>
[CmdletBinding()]
param(
    [ValidateSet('all', 'mic', 'speakers', 'engines')]
    [string]$Only = 'all',
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
try { chcp 65001 | Out-Null } catch { }

$Root   = Split-Path -Parent $PSScriptRoot
$PyDir  = Join-Path $Root '_tools\win-python'
$PyExe  = Join-Path $PyDir 'python.exe'
$Models = Join-Path $Root '_tools\models'

function Say([string]$text, [string]$colour = 'Gray') {
    Write-Host $text -ForegroundColor $colour
}

if (-not (Test-Path $PyExe)) {
    Say "Портативного Python нет в _tools\win-python." Yellow
    Say "Сначала запустите scripts\check-keys.ps1 - он его принесёт." Yellow
    exit 1
}

# ----------------------------------------------------------------------
# движки и модели
# ----------------------------------------------------------------------

$needsModels = ($Only -eq 'all' -or $Only -eq 'engines')

if (-not $SkipInstall -and $needsModels) {
    Say "`n[1/3] Движки речи" Cyan
    # piper-phonemize сюда не входит намеренно: колёс под windows нет ни для
    # 3.11, ни для 3.12, установка падает на сборке из исходников. Проверка
    # piper останется пропущенной, это ожидаемо.
    $packages = @(
        'vosk==0.3.45',
        'faster-whisper==1.1.1',
        'openwakeword==0.6.0',
        'onnxruntime',
        'scipy==1.14.1',
        'scikit-learn==1.5.2'
    )
    & $PyExe -m pip install --disable-pip-version-check --no-input @packages
    if ($LASTEXITCODE -ne 0) { Say "pip не смог - смотрите вывод выше." Red; exit 1 }

    Say "`n[2/3] Модели" Cyan
    New-Item -ItemType Directory -Force -Path $Models | Out-Null

    $voskDir = Join-Path $Models 'vosk-model-small-ru-0.22'
    if (-not (Test-Path $voskDir)) {
        $zip = Join-Path $Models 'vosk-ru-small.zip'
        Say "  vosk-model-small-ru-0.22 (45 МБ)..."
        # Зеркало на huggingface, а не alphacephei.com: оригинал отдаёт
        # ~500 байт/с, качается сутки.
        Invoke-WebRequest -UseBasicParsing -OutFile $zip `
            'https://huggingface.co/rhasspy/vosk-models/resolve/main/ru/vosk-model-small-ru-0.22.zip'
        Expand-Archive -Path $zip -DestinationPath $Models -Force
        Remove-Item $zip -Force
    } else { Say "  vosk-model-small-ru-0.22 уже на месте" }

    # faster-whisper tiny: CTranslate2-экспорт, качается сам при первом обращении,
    # но тогда он лёг бы в кэш пользователя - тянем в папку проекта.
    $whisperDir = Join-Path $Models 'faster-whisper-tiny'
    if (-not (Test-Path (Join-Path $whisperDir 'model.bin'))) {
        Say "  faster-whisper-tiny (75 МБ)..."
        New-Item -ItemType Directory -Force -Path $whisperDir | Out-Null
        $hf = 'https://huggingface.co/Systran/faster-whisper-tiny/resolve/main'
        foreach ($f in @('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt')) {
            Invoke-WebRequest -UseBasicParsing -OutFile (Join-Path $whisperDir $f) "$hf/$f"
        }
    } else { Say "  faster-whisper-tiny уже на месте" }

    $owwDir = Join-Path $Models 'openwakeword'
    New-Item -ItemType Directory -Force -Path $owwDir | Out-Null
    $base = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1'
    foreach ($f in @('melspectrogram.onnx', 'embedding_model.onnx', 'silero_vad.onnx', 'hey_jarvis_v0.1.onnx')) {
        $dest = Join-Path $owwDir $f
        if (-not (Test-Path $dest)) {
            Say "  openwakeword/$f..."
            Invoke-WebRequest -UseBasicParsing -OutFile $dest "$base/$f"
        }
    }
    # Библиотека ищет мелспектрограмму и эмбеддинги рядом с собой, а не в
    # папке моделей - кладём копию туда, иначе Model() их не найдёт.
    $res = & $PyExe -c "import openwakeword,os;print(os.path.join(os.path.dirname(openwakeword.__file__),'resources','models'))"
    New-Item -ItemType Directory -Force -Path $res | Out-Null
    foreach ($f in @('melspectrogram.onnx', 'embedding_model.onnx', 'silero_vad.onnx')) {
        Copy-Item (Join-Path $owwDir $f) (Join-Path $res $f) -Force
    }
    # Файл называется hey_jarvis_v0.1, а движок ищет его по тексту фразы.
    Copy-Item (Join-Path $owwDir 'hey_jarvis_v0.1.onnx') (Join-Path $owwDir 'hey_jarvis.onnx') -Force
}

# ----------------------------------------------------------------------
# прогон
# ----------------------------------------------------------------------

Say "`n[3/3] Тесты" Cyan
$env:AYRIS_TEST_STT_MODEL     = Join-Path $Models 'vosk-model-small-ru-0.22'
$env:AYRIS_TEST_WHISPER_MODEL = Join-Path $Models 'faster-whisper-tiny'
$env:AYRIS_TEST_WAKE_MODELS   = Join-Path $Models 'openwakeword'
# Тест обрыва звука сам себя пропускает, пока эта переменная не выставлена:
# на машине без колонок он бы прошёл впустую.
$env:AYRIS_TEST_SPEAKERS      = '1'

$selector = switch ($Only) {
    'mic'      { @('tests/unit/test_vad.py::TestLiveMicrophone') }
    'speakers' { @('tests/unit/test_tts_player.py::TestCancellation') }
    'engines'  { @('tests/unit/test_stt_offline.py', 'tests/unit/test_wake_word.py') }
    default    { @('tests/unit') }
}

if ($Only -eq 'all' -or $Only -eq 'mic') {
    Say ""
    Say "Сейчас будет замер шума комнаты: две секунды тишины." Yellow
    Say "Нажмите Enter, когда готовы помолчать." Yellow
    [void](Read-Host)
}

if ($Only -eq 'all' -or $Only -eq 'speakers') {
    Say ""
    Say "Потом будет короткий тихий гудок в колонках - так и надо." Yellow
}

$code = 1
Push-Location $Root
try {
    $basetemp = Join-Path $env:TEMP ("ayris_pt_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    & $PyExe -m pytest -m hardware -v --basetemp=$basetemp -p no:cacheprovider @selector
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

Say ""
if ($code -eq 0) {
    Say "Готово: всё зелёное." Green
} else {
    Say "Что-то упало - пришлите вывод выше целиком." Red
}
Say "Пропуск с пометкой piper - ожидаемо: колёс piper-phonemize под Windows нет."
exit $code
