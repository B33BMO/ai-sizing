# Start the memlm job pack on a Windows machine with Docker Desktop (WSL2 backend) and NVIDIA GPUs.
#   powershell -ExecutionPolicy Bypass -File run-all.ps1 -Jobs jobs\tonight.jobs
# One read-only Spero tool container + one training container per GPU. Results/logs land in .\out\<job>\.
param([string]$Jobs = "jobs\tonight.jobs", [string]$Out = "$PSScriptRoot\out", [string]$Hf = "$PSScriptRoot\hf-cache")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out, $Hf | Out-Null
docker network inspect memlm *> $null; if ($LASTEXITCODE -ne 0) { docker network create memlm | Out-Null }
docker rm -f memlm-kb *> $null
docker run -d --name memlm-kb --network memlm --restart unless-stopped memlm-kb:latest | Out-Null
Write-Host "tool service started (memlm-kb)"
Get-Content $Jobs | Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*#' } | ForEach-Object {
    $gpu, $name, $cmd = ($_ -split '\|', 3) | ForEach-Object { $_.Trim() }
    New-Item -ItemType Directory -Force -Path "$Out\$name" | Out-Null
    docker rm -f "memlm-$name" *> $null
    docker run -d --name "memlm-$name" --gpus "device=$gpu" --network memlm --shm-size 16g `
        -v "${Out}:/out" -v "${Hf}:/hf" memlm-train:latest `
        bash -lc "($cmd) > /out/$name/job.log 2>&1; echo EXIT=`$? >> /out/$name/job.log" | Out-Null
    Write-Host "GPU $gpu -> $name"
}
Write-Host "`nwatch: powershell -File status.ps1"
