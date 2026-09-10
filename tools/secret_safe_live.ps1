# Secret-safe live checks against an isolated cline-proxy.
# Reads the real config locally to set process env; never prints secrets.
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,
    [int]$MaxTokens = 16
)

$ErrorActionPreference = "Stop"
$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$env:CLINE_PROXY_GATEWAY_KEY = [string]$cfg.server.api_key
if ([string]::IsNullOrWhiteSpace($env:CLINE_PROXY_GATEWAY_KEY)) {
    Write-Output "FAILED: gateway key missing from config (not printed)"
    exit 1
}

function Invoke-Safe([string]$Name, [scriptblock]$Body) {
    try {
        $result = & $Body
        Write-Output $result
    }
    catch {
        Write-Output ("FAILED {0}: {1}" -f $Name, $_.Exception.Message)
        exit 1
    }
}

Invoke-Safe "healthz" {
    $r = Invoke-WebRequest -Uri "$BaseUrl/healthz" -UseBasicParsing
    "healthz status=$($r.StatusCode) bytes=$($r.RawContentLength)"
}

Invoke-Safe "readyz" {
    $r = Invoke-WebRequest -Uri "$BaseUrl/readyz" -UseBasicParsing
    "readyz status=$($r.StatusCode)"
}

$headers = @{
    "x-api-key"         = $env:CLINE_PROXY_GATEWAY_KEY
    "anthropic-version" = "2023-06-01"
    "content-type"      = "application/json"
}

$streamBody = @{
    model      = "claude-sonnet-4-6"
    max_tokens = $MaxTokens
    stream     = $true
    messages   = @(@{ role = "user"; content = "Reply with the single word OK." })
} | ConvertTo-Json -Compress

Invoke-Safe "stream" {
    $r = Invoke-WebRequest -Uri "$BaseUrl/v1/messages" -Method POST -Headers $headers -Body $streamBody -UseBasicParsing
    $text = $r.Content
    $hasText = $text -match "content_block_delta"
    $hasError = $text -match '"type":"error"'
    $hasStop = $text -match "message_stop"
    "stream status=$($r.StatusCode) has_delta=$hasText has_error=$hasError has_stop=$hasStop bytes=$($text.Length)"
}

$jsonBody = @{
    model      = "claude-sonnet-4-6"
    max_tokens = $MaxTokens
    stream     = $false
    messages   = @(@{ role = "user"; content = "Reply with the single word OK." })
} | ConvertTo-Json -Compress

Invoke-Safe "nonstream" {
    $r = Invoke-WebRequest -Uri "$BaseUrl/v1/messages" -Method POST -Headers $headers -Body $jsonBody -UseBasicParsing
    $parsed = $r.Content | ConvertFrom-Json
    $blocks = @($parsed.content).Count
    $stop = [string]$parsed.stop_reason
    $inTok = $parsed.usage.input_tokens
    $outTok = $parsed.usage.output_tokens
    "nonstream status=$($r.StatusCode) blocks=$blocks stop=$stop input_tokens=$inTok output_tokens=$outTok"
}

Write-Output "DONE"
Remove-Item Env:CLINE_PROXY_GATEWAY_KEY -ErrorAction SilentlyContinue
