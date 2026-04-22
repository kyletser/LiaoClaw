<#
.SYNOPSIS
LiaoClaw Windows 本地调试启动脚本
.DESCRIPTION
使用方式:
   liaoclaw-web                    # web模式，浏览器访问 http://127.0.0.1:8787
  .\dev.ps1 -Mode cli                   # CLI 交互模式
  .\dev.ps1 -Mode im -Transport longconn  # 飞书长连接模式
#>
param(
    [ValidateSet("im", "cli", "web")]
    [string]$Mode       = "im",
    [string]$Transport  = "webhook",
    [string]$ListenHost = "127.0.0.1",
    [int]   $Port       = 8787,
    [string]$Workspace  = ".",
    [string]$LogLevel   = "debug"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptRoot) {
    $scriptRoot = (Get-Location).Path
}

# 加载 .env.ps1（如果存在）
$envFile = Join-Path $scriptRoot ".env.ps1"
if (Test-Path $envFile) {
    Write-Host "[dev] Loading $envFile ..." -ForegroundColor Cyan
    . $envFile
}

# 确保在项目根目录
Set-Location $scriptRoot

# 统一使用虚拟环境 Python（若存在），避免 python/pip 指向不一致。
$pythonExe = "python"
if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    $pythonExe = (Resolve-Path -LiteralPath ".\.venv\Scripts\python.exe").Path
}

# 确保已安装（开发模式）
& $pythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('coding_agent') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[dev] Installing liaoclaw in editable mode ..." -ForegroundColor Yellow
    & $pythonExe -m pip install -e ".[dev]"
}

$provider = if ($env:LIAOCLAW_PROVIDER) { $env:LIAOCLAW_PROVIDER } else { "anthropic" }
$modelId  = if ($env:LIAOCLAW_MODEL_ID) { $env:LIAOCLAW_MODEL_ID } else { "glm-4.5-air" }

if ($Mode -eq "im") {
    $appId     = $env:FEISHU_APP_ID
    $appSecret = $env:FEISHU_APP_SECRET
    $verifyTk  = $env:FEISHU_VERIFY_TOKEN

    if (-not $appId -or -not $appSecret) {
        Write-Host "[dev] ERROR: FEISHU_APP_ID and FEISHU_APP_SECRET must be set." -ForegroundColor Red
        Write-Host "[dev] Create .env.ps1 from .env.ps1.example and fill in values." -ForegroundColor Red
        exit 1
    }

    $pyArgs = @(
        "-m", "im",
        "--platform", "feishu",
        "--transport", $Transport,
        "--workspace", $Workspace,
        "--host", $ListenHost,
        "--port", $Port,
        "--provider", $provider,
        "--model-id", $modelId,
        "--feishu-app-id", $appId,
        "--feishu-app-secret", $appSecret,
        "--log-level", $LogLevel
    )
    if ($verifyTk) {
        $pyArgs += @("--feishu-verify-token", $verifyTk)
    }

    Write-Host "[dev] Starting IM service ($Transport) on ${ListenHost}:${Port} ..." -ForegroundColor Green
    Write-Host "[dev] Provider: $provider | Model: $modelId" -ForegroundColor Green
    & $pythonExe @pyArgs

} elseif ($Mode -eq "cli") {
    $pyArgs = @(
        "-m", "coding_agent",
        "--mode", "interactive",
        "--workspace", $Workspace,
        "--provider", $provider,
        "--model-id", $modelId
    )

    Write-Host "[dev] Starting CLI interactive mode ..." -ForegroundColor Green
    Write-Host "[dev] Provider: $provider | Model: $modelId" -ForegroundColor Green
    & $pythonExe @pyArgs
    
} else {
    $pyArgs = @(
        "-m", "coding_agent.web",
        "--workspace", $Workspace,
        "--provider", $provider,
        "--model-id", $modelId,
        "--host", $ListenHost,
        "--port", $Port
    )

    Write-Host "[dev] Starting Web mode on ${ListenHost}:${Port} ..." -ForegroundColor Green
    Write-Host "[dev] Provider: $provider | Model: $modelId" -ForegroundColor Green
    & $pythonExe @pyArgs
}
