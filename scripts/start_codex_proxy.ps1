param(
    [ValidateRange(1, 65535)]
    [int]$Port = 7892
)
$ErrorActionPreference = 'Stop'
$proxyAddress = "http://127.0.0.1:$Port"
$probe = [System.Net.Sockets.TcpClient]::new()
try {
    $connection = $probe.ConnectAsync('127.0.0.1', $Port)
    if (-not $connection.Wait(3000)) { throw 'Connection timed out.' }
} catch {
    throw "Local proxy 127.0.0.1:$Port is unavailable. Start your proxy application and verify its port."
} finally {
    $probe.Dispose()
}
$package = Get-AppxPackage -Name OpenAI.Codex | Select-Object -First 1
if (-not $package) { throw 'Codex desktop package was not found.' }
$appExe = Join-Path $package.InstallLocation 'app\ChatGPT.exe'
if (-not (Test-Path -LiteralPath $appExe)) { throw 'Codex desktop executable was not found.' }
$env:HTTP_PROXY = $proxyAddress
$env:HTTPS_PROXY = $proxyAddress
$env:ALL_PROXY = $proxyAddress
$env:NO_PROXY = 'localhost,127.0.0.1,::1'
Start-Process -FilePath $appExe -ArgumentList "--proxy-server=$proxyAddress" -WindowStyle Hidden
