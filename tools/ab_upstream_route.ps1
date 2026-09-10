# Secret-safe A/B: isolated cline-proxy, real config, tiny Anthropic turns.
# Prints only timings/status/route. Never prints secrets or completion text.
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$Exe,
    [Parameter(Mandatory = $true)]
    [string]$WorkDir,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [Parameter(Mandatory = $true)]
    [string]$Route, # direct | socks5://127.0.0.1:10888
    [int]$Samples = 4,
    [int]$MaxTokens = 8
)

$ErrorActionPreference = "Stop"
$logs = Join-Path $WorkDir "logs"
$state = Join-Path $WorkDir "runtime-state.json"
$console = Join-Path $WorkDir "console.err"
New-Item -ItemType Directory -Force -Path $WorkDir, $logs | Out-Null

$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$env:CLINE_PROXY_GATEWAY_KEY = [string]$cfg.server.api_key
if ([string]::IsNullOrWhiteSpace($env:CLINE_PROXY_GATEWAY_KEY)) {
    Write-Output "FAILED: gateway key missing"
    exit 1
}

$proc = Start-Process -FilePath $Exe -ArgumentList @(
    "--config", $Config,
    "--bind", "127.0.0.1:$Port",
    "--state-file", $state,
    "--log-directory", $logs,
    "--no-color",
    "--upstream-proxy", $Route
) -RedirectStandardError $console -RedirectStandardOutput (Join-Path $WorkDir "console.out") -PassThru -WindowStyle Hidden

try {
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 250
        try {
            $h = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -UseBasicParsing -TimeoutSec 2
            if ($h.StatusCode -eq 200) { $ok = $true; break }
        } catch {}
    }
    if (-not $ok) {
        Write-Output "FAILED: proxy did not become ready"
        exit 1
    }

    $headers = @{
        "x-api-key"         = $env:CLINE_PROXY_GATEWAY_KEY
        "anthropic-version" = "2023-06-01"
        "content-type"      = "application/json"
    }
    $body = (@{
            model      = "claude-sonnet-4-6"
            max_tokens = $MaxTokens
            stream     = $true
            messages   = @(@{ role = "user"; content = "Reply with the single word OK." })
        } | ConvertTo-Json -Compress)

    $rows = @()
    for ($n = 0; $n -lt $Samples; $n++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $status = 0
        $err = $false
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/messages" -Method POST -Headers $headers -Body $body -UseBasicParsing -TimeoutSec 120
            $status = [int]$r.StatusCode
            $err = $r.Content -match '"type":"error"'
        }
        catch {
            $status = 0
            $err = $true
        }
        $sw.Stop()
        $rows += [pscustomobject]@{
            sample      = $n
            status      = $status
            error_event = $err
            client_ms   = [int]$sw.Elapsed.TotalMilliseconds
        }
        Write-Output ("sample={0} status={1} error_event={2} client_ms={3}" -f $n, $status, $err, [int]$sw.Elapsed.TotalMilliseconds)
    }
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 400
    }
    Remove-Item Env:CLINE_PROXY_GATEWAY_KEY -ErrorAction SilentlyContinue
}

Write-Output "DONE"
