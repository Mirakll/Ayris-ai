$ErrorActionPreference = 'SilentlyContinue'
$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$phys = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1024 / 1024, 1)
$usedGB = [math]::Round($phys - $freeGB, 1)

Write-Output ("Physical RAM (GB): " + $phys)
Write-Output ("Used (GB):         " + $usedGB)
Write-Output ("Free (GB):         " + $freeGB)
Write-Output "----- breakdown (MB) -----"

$paths = @(
  '\Memory\Cache Bytes',
  '\Memory\Pool Nonpaged Bytes',
  '\Memory\Pool Paged Resident Bytes',
  '\Memory\Modified Page List Bytes',
  '\Memory\Standby Cache Normal Priority Bytes',
  '\Memory\Standby Cache Reserve Bytes',
  '\Memory\Standby Cache Core Bytes',
  '\Memory\Committed Bytes',
  '\Memory\Commit Limit'
)
$c = Get-Counter -Counter $paths
foreach ($s in $c.CounterSamples) {
  $name = $s.Path.Split('\')[-1]
  $mb = [math]::Round($s.CookedValue / 1MB)
  Write-Output ($name.PadRight(42) + " = " + $mb + " MB")
}

Write-Output "----- top 15 processes by working set (MB) -----"
Get-Process |
  Sort-Object WorkingSet64 -Descending |
  Select-Object -First 15 Name, @{N='WS_MB'; E={[math]::Round($_.WorkingSet64 / 1MB)}}, Id |
  Format-Table -AutoSize | Out-String | Write-Output

$sum = (Get-Process | Measure-Object WorkingSet64 -Sum).Sum
Write-Output ("Sum of ALL process working sets (GB): " + [math]::Round($sum / 1GB, 1))
