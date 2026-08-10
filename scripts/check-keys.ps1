#Requires -Version 5.1
<#
    Ayris - проверка облачных ключей распознавания речи.

    Зачем обёртка: на этой машине нет Python, а песочница Claude не может
    достучаться до Яндекса, Google, Azure и OpenAI - их домены не в allowlist.
    Да и ключ не должен уезжать с машины владельца. Поэтому здесь приносится
    портативный CPython 3.12 (embeddable-сборка с python.org, ~11 МБ, ставить
    в систему ничего не надо), к нему распаковываются колёса из _wheels\, и
    запускается scripts\check_cloud_keys.py.

    Всё складывается в _tools\win-python\ - папка под .gitignore, удаляется
    обычным Remove-Item, в реестр и в PATH ничего не пишется.

    Запуск (одной строкой в PowerShell):
      powershell -ExecutionPolicy Bypass -File "E:\мистер бит ест рис\scripts\check-keys.ps1"

    Ключи:
      -Providers yandex,openai   проверить не всех, а перечисленных
      -Wav <путь>               взять готовый WAV вместо микрофона
      -Seconds 8                сколько писать с микрофона
      -Force                    спросить ключи заново, даже если уже сохранены
      -Reinstall                выкинуть _tools\win-python и собрать заново
#>
[CmdletBinding()]
param(
    [string]$Providers = 'all',
    [string]$Wav = '',
    [double]$Seconds = 5,
    [switch]$Force,
    [switch]$Reinstall
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # иначе Invoke-WebRequest тормозит на больших файлах
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
try { chcp 65001 | Out-Null } catch { }

Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root    = Split-Path -Parent $PSScriptRoot
$Wheels  = Join-Path $Root '_wheels'
$PyDir   = Join-Path $Root '_tools\win-python'
$Lib     = Join-Path $PyDir 'lib'
$PyExe   = Join-Path $PyDir 'python.exe'
$Script  = Join-Path $PSScriptRoot 'check_cloud_keys.py'

function Write-Step { param($t) Write-Host ''; Write-Host $t -ForegroundColor Yellow }
function Write-Ok   { param($t) Write-Host "      $t" -ForegroundColor Green }
function Write-Dim  { param($t) Write-Host "      $t" -ForegroundColor DarkGray }
function Write-Bad  { param($t) Write-Host "      $t" -ForegroundColor Red }

function Remove-File {
    # Windows подставляет %TEMP% в коротком виде 8.3, если имя профиля не
    # латиницей: C:\Users\328F~1\AppData\Local\Temp. Скачать и распаковать по
    # такому пути получается - это делает .NET, - а Remove-Item сначала гоняет
    # путь через разрешение шаблонов, спотыкается о «~» и заявляет, что объекта
    # нет. Ошибка терминирующая: её не гасит -ErrorAction SilentlyContinue.
    # Поэтому удаляем через .NET, буквальным путём.
    param([string]$Path)
    try { [System.IO.File]::Delete($Path) } catch { }
}

function Remove-Tree {
    # То же самое для каталога: -Recurse -Force спотыкается там же.
    param([string]$Path)
    try { [System.IO.Directory]::Delete($Path, $true) } catch { }
}

if (-not (Test-Path $Script)) { throw "Не найден $Script" }
if ($Reinstall -and (Test-Path $PyDir)) { Remove-Tree $PyDir }

Write-Host ''
Write-Host "Проект: $Root" -ForegroundColor Cyan

# =====================================================================
#  Пакеты, которые нужны скрипту проверки
# =====================================================================
#   n   - имя на PyPI
#   tag - тег колеса. Для чистого Python это py3-none-any, для cffi и
#         sounddevice нужны сборки под Windows: они тащат за собой .pyd и DLL
#         (в sounddevice внутри лежит PortAudio, отдельно его ставить не надо).
$Deps = @(
    @{ n = 'httpx';             tag = 'py3-none-any' }
    @{ n = 'httpcore';          tag = 'py3-none-any' }
    @{ n = 'h11';               tag = 'py3-none-any' }
    @{ n = 'anyio';             tag = 'py3-none-any' }
    @{ n = 'sniffio';           tag = 'py3-none-any' }
    @{ n = 'idna';              tag = 'py3-none-any' }
    @{ n = 'certifi';           tag = 'py3-none-any' }
    @{ n = 'typing-extensions'; tag = 'py3-none-any' }
    @{ n = 'keyring';           tag = 'py3-none-any' }
    @{ n = 'jaraco.classes';    tag = 'py3-none-any' }
    @{ n = 'jaraco.context';    tag = 'py3-none-any' }
    @{ n = 'jaraco.functools';  tag = 'py3-none-any' }
    @{ n = 'more-itertools';    tag = 'py3-none-any' }
    @{ n = 'pywin32-ctypes';    tag = 'py3-none-any' }
    @{ n = 'sounddevice';       tag = 'py3-none-win_amd64' }
    @{ n = 'cffi';              tag = 'cp312-cp312-win_amd64' }
    @{ n = 'pycparser';         tag = 'py3-none-any' }
)

# Версии embeddable-сборки в порядке предпочтения. 3.12.x, потому что колёса в
# _wheels\ собраны под cp312. После 3.12.10 ветка ушла в security-only режим,
# и бинарных сборок для неё больше не выпускают - поэтому список, а не одна.
$PyVersions = @('3.12.10', '3.12.9', '3.12.8', '3.12.7', '3.12.6')

# =====================================================================
#  Python 3.12 для Windows
# =====================================================================

function Get-Python {
    # Последняя 3.12 с бинарными сборками заранее неизвестна, поэтому версии
    # примеряются по списку: первая, которая скачалась, и берётся.
    Write-Step 'Портативный CPython 3.12'
    foreach ($v in $PyVersions) {
        $url = "https://www.python.org/ftp/python/$v/python-$v-embed-amd64.zip"
        $tmp = Join-Path $env:TEMP "python-$v-embed-amd64.zip"
        try {
            Write-Dim "качаю $url"
            Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
        } catch {
            Write-Dim "нет: $($_.Exception.Message)"
            Remove-File $tmp
            continue
        }
        New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
        Expand-Archive -Path $tmp -DestinationPath $PyDir -Force
        Remove-File $tmp
        return
    }
    throw 'Не удалось скачать CPython 3.12 с python.org. Проверьте подключение к интернету.'
}

function Find-Wheel {
    param($Name, $Tag)
    # В именах файлов PyPI заменяет дефисы и точки на подчёркивания, но не всегда:
    # рядом лежат и jaraco.classes-3.4.0-..., и jaraco_context-6.1.2-...
    $variants = @($Name, ($Name -replace '-', '_'), ($Name -replace '\.', '_'),
                  ($Name -replace '[-.]', '_')) | Select-Object -Unique
    foreach ($variant in $variants) {
        $found = Get-ChildItem -Path $Wheels -Filter "$variant-*-$Tag.whl" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

function Get-Wheel {
    # Колеса нет в _wheels\ - принести с PyPI. Штатно не срабатывает: setup-tools.ps1
    # кладёт туда всё, что нужно проекту, включая сборки под Windows.
    param($Name, $Tag)
    $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$Name/json" -UseBasicParsing
    $file = $meta.urls | Where-Object { $_.filename -like "*-$Tag.whl" } | Select-Object -First 1
    if (-not $file) { throw "На PyPI нет колеса $Name с тегом $Tag" }
    $target = Join-Path $Wheels $file.filename
    Invoke-WebRequest -Uri $file.url -OutFile $target -UseBasicParsing
    $actual = (Get-FileHash -Path $target -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $file.digests.sha256) {
        Remove-File $target
        throw "$($file.filename): sha256 не совпал"
    }
    Write-Dim "скачано $($file.filename)"
    return $target
}

function Expand-Wheel {
    # Колесо - обычный zip, и «установка» чистого Python-пакета это распаковка.
    # Всё в один каталог: он и попадает в sys.path через python312._pth.
    param($Path)
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        foreach ($entry in $zip.Entries) {
            $target = Join-Path $Lib $entry.FullName
            if ($entry.FullName.EndsWith('/')) { continue }
            $parent = Split-Path -Parent $target
            if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    } finally {
        $zip.Dispose()
    }
}

# =====================================================================
#  Сборка окружения
# =====================================================================

if (-not (Test-Path $PyExe)) {
    Get-Python
    Write-Ok "распакован в $PyDir"
} else {
    Write-Step 'Портативный CPython 3.12'
    Write-Ok 'уже на месте'
}

# Embeddable-сборка живёт в изолированном режиме: sys.path задаётся строками из
# python312._pth, а PYTHONPATH при этом игнорируется. Поэтому каталог с колёсами
# и папку src\ прописываем прямо в этот файл. Пути относительные - от каталога
# с python.exe, так что папку проекта можно перенести.
$pth = Get-ChildItem -Path $PyDir -Filter 'python*._pth' | Select-Object -First 1
if (-not $pth) { throw "В $PyDir нет файла python*._pth - сборка не та." }
$lines = @(
    (Get-ChildItem -Path $PyDir -Filter 'python*.zip' | Select-Object -First 1).Name
    '.'
    'lib'
    '..\..\src'
    'import site'
)
Set-Content -Path $pth.FullName -Value $lines -Encoding ASCII

Write-Step 'Зависимости из _wheels'
New-Item -ItemType Directory -Force -Path $Lib, $Wheels | Out-Null
$missing = @()
foreach ($dep in $Deps) {
    $wheel = Find-Wheel -Name $dep.n -Tag $dep.tag
    if (-not $wheel) {
        try {
            $wheel = Get-Wheel -Name $dep.n -Tag $dep.tag
        } catch {
            Write-Bad "$($dep.n): $($_.Exception.Message)"
            $missing += $dep.n
            continue
        }
    }
    Expand-Wheel -Path $wheel
}
if ($missing.Count) {
    Write-Bad "не хватает: $($missing -join ', ')"
    if ($missing -contains 'httpx' -or $missing -contains 'keyring') {
        throw 'Без httpx и keyring проверка невозможна.'
    }
    Write-Dim 'микрофон будет недоступен, скрипт возьмёт WAV-фикстуру'
}
Write-Ok "$($Deps.Count - $missing.Count) пакетов в $Lib"

Write-Step 'Проверка окружения'
# keyring не публикует __version__, поэтому спрашиваем у него бэкенд: заодно
# проверяется, что подтянулись jaraco.* и pywin32-ctypes.
$probe = & $PyExe -c "import httpx, ayris; from keyring.backends.Windows import WinVaultKeyring; print(f'httpx {httpx.__version__}, ayris {ayris.__version__}, WinVault {WinVaultKeyring.viable}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Bad ($probe -join [Environment]::NewLine)
    throw 'Портативный Python не смог импортировать httpx/keyring/ayris.'
}
Write-Ok "$probe"

# =====================================================================
#  Собственно проверка
# =====================================================================

Write-Step 'Ключи и живые запросы'
Write-Dim 'ключ вводится скрыто и уходит в диспетчер учётных данных Windows'
Write-Host ''

$pyArgs = @($Script, '--providers', $Providers, '--seconds', $Seconds.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($Wav)   { $pyArgs += @('--wav', $Wav) }
if ($Force) { $pyArgs += '--force' }

# Ключ не передаётся аргументом: он бы остался в истории PowerShell и в списке
# процессов. Скрипт спрашивает его сам через getpass.
& $PyExe @pyArgs
$code = $LASTEXITCODE

Write-Host ''
if ($code -eq 0) {
    Write-Host 'Все выбранные сервисы ответили.' -ForegroundColor Green
} else {
    Write-Host 'Часть сервисов не ответила - подробности выше.' -ForegroundColor Red
}
Write-Host 'Пришлите Claude вывод целиком: ключей в нём нет, только маски и ошибки.' -ForegroundColor Cyan
exit $code
