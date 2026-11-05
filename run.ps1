<#
.SYNOPSIS
    kartOS Master Automation & Launch Script for Windows PowerShell
#>
param (
    [string]$Command = "menu",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$PiperModel = "en_US-lessac-medium.onnx"
$PiperConfig = "en_US-lessac-medium.onnx.json"
$PiperModelUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
$PiperConfigUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "               kartOS Voice Runtime               " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Virtual Environment Setup
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

# 2. Dependency Check & Installation
Write-Host "[*] Checking dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install -q -r requirements.txt

# 3. Environment Variable Check
if (Test-Path ".env") {
    Write-Host "[*] Reading .env file..." -ForegroundColor Yellow
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and ($line -match "^([^=]+)=(.*)$")) {
            $name = $matches[1].Trim()
            $val = $matches[2].Trim().Trim('"').Trim("'")
            if (-not [System.Environment]::GetEnvironmentVariable($name)) {
                [System.Environment]::SetEnvironmentVariable($name, $val, "Process")
            }
        }
    }
}

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "[!] Warning: ANTHROPIC_API_KEY is not set." -ForegroundColor DarkYellow
    Write-Host "    (Live Anthropic LLM requires this key. Mock/demo/test modes work without it.)"
}

# 4. Piper Model Downloader
function Download-PiperModel {
    Write-Host "[*] Checking Piper voice model files..." -ForegroundColor Yellow
    if (-not (Test-Path $PiperModel)) {
        Write-Host "[*] Downloading Piper voice model ($PiperModel)..." -ForegroundColor Green
        Invoke-WebRequest -Uri $PiperModelUrl -OutFile $PiperModel
    }
    if (-not (Test-Path $PiperConfig)) {
        Write-Host "[*] Downloading Piper voice config ($PiperConfig)..." -ForegroundColor Green
        Invoke-WebRequest -Uri $PiperConfigUrl -OutFile $PiperConfig
    }
    Write-Host "[+] Piper voice model ready: $PiperModel" -ForegroundColor Green
}

# 5. Command Execution
switch ($Command.ToLower()) {
    "test" {
        Write-Host "[*] Running test suite..." -ForegroundColor Cyan
        & $VenvPython -m pytest -q $RemainingArgs
    }
    "eval" {
        Write-Host "[*] Running evaluation harness..." -ForegroundColor Cyan
        & $VenvPython -m kartos.eval $RemainingArgs
    }
    "demo" {
        Write-Host "[*] Running simulated conversation demo..." -ForegroundColor Cyan
        & $VenvPython main.py --demo $RemainingArgs
    }
    "web" {
        Write-Host "[*] Launching browser WebSocket server..." -ForegroundColor Cyan
        & $VenvPython -m kartos.server $RemainingArgs
    }
    "server" {
        Write-Host "[*] Launching browser WebSocket server..." -ForegroundColor Cyan
        & $VenvPython -m kartos.server $RemainingArgs
    }
    "live" {
        Download-PiperModel
        if (-not $env:ANTHROPIC_API_KEY) {
            Write-Host "[!] Error: ANTHROPIC_API_KEY is required to run live." -ForegroundColor Red
            exit 1
        }
        Write-Host "[*] Starting live voice agent..." -ForegroundColor Cyan
        & $VenvPython main.py --piper-model $PiperModel $RemainingArgs
    }
    "setup" {
        Download-PiperModel
        Write-Host "[+] Setup completed successfully." -ForegroundColor Green
    }
    default {
        Write-Host "`nSelect an option:"
        Write-Host "  1) Run Test Suite (pytest -q)"
        Write-Host "  2) Run Evaluation Harness (python -m kartos.eval)"
        Write-Host "  3) Run Demo Simulation (main.py --demo --speed 4)"
        Write-Host "  4) Launch Web Server (http://localhost:8080)"
        Write-Host "  5) Run Live Agent (Microphone + Piper + Claude)"
        Write-Host "  6) Download Voice Models (Piper ONNX)"
        Write-Host "  q) Quit`n"
        $choice = Read-Host "Enter choice [1-6, q]"
        switch ($choice) {
            "1" { & $VenvPython -m pytest -q }
            "2" { & $VenvPython -m kartos.eval --speed 4 }
            "3" { & $VenvPython main.py --demo --speed 4 }
            "4" { & $VenvPython -m kartos.server }
            "5" {
                Download-PiperModel
                & $VenvPython main.py --piper-model $PiperModel
            }
            "6" { Download-PiperModel }
            default { Write-Host "Exiting." }
        }
    }
}
