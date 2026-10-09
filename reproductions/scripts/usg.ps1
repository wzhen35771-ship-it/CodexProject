param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("check", "install", "smoke", "train-psg", "eval-psg", "train-joint", "eval-joint")]
    [string]$Action,

    [string]$Checkpoint = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$UsgRoot = Join-Path $Root "USG"

function Enter-Usg {
    if (!(Test-Path $UsgRoot)) {
        throw "USG checkout not found at $UsgRoot"
    }
    Set-Location $UsgRoot
}

function Check-Command {
    param([string]$Name)
    if (!(Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing command: $Name"
    }
}

function Show-Status {
    Enter-Usg
    Check-Command $Python

    Write-Host "USG root: $UsgRoot"
    Write-Host "Git commit:"
    git log -1 --oneline

    Write-Host ""
    Write-Host "Python:"
    & $Python --version

    Write-Host ""
    Write-Host "Required dataset paths for PSG config:"
    Get-Content "configs\psg.yaml" | Select-String "ann:|image_root:"

    Write-Host ""
    Write-Host "Quick import check:"
    & $Python -c "import sys; print(sys.executable); import yaml, numpy, torch, sng_parser; print('basic imports ok'); print('cuda', torch.cuda.is_available())"
}

function Install-Dependencies {
    Enter-Usg
    & $Python -m pip install torch torchvision numpy einops scipy pyyaml pytest tqdm open_clip_torch timm av plyfile matplotlib SceneGraphParser spacy
    & $Python -m spacy download en_core_web_sm
}

function Run-Smoke {
    Enter-Usg
    & $Python (Join-Path $Root "scripts\usg_smoke.py")
}

function Train-Psg {
    Enter-Usg
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $Python "train.py" --config "configs\psg.yaml"
}

function Eval-Psg {
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        throw "Pass -Checkpoint, for example: -Checkpoint checkpoints\psg\best.pt"
    }
    Enter-Usg
    & $Python "eval.py" --config "configs\psg.yaml" --ckpt $Checkpoint --task sgdet
}

function Train-Joint {
    Enter-Usg
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $Python "train_joint.py" --config "configs\joint.yaml" --stage both
}

function Eval-Joint {
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        throw "Pass -Checkpoint, for example: -Checkpoint checkpoints\joint\stage2.pt"
    }
    Enter-Usg
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $Python "eval_joint.py" --config "configs\joint.yaml" --ckpt $Checkpoint
}

switch ($Action) {
    "check" { Show-Status }
    "install" { Install-Dependencies }
    "smoke" { Run-Smoke }
    "train-psg" { Train-Psg }
    "eval-psg" { Eval-Psg }
    "train-joint" { Train-Joint }
    "eval-joint" { Eval-Joint }
}
