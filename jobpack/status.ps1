# Show each job's container state and the last lines of its log.
param([string]$Out = "$PSScriptRoot\out")
docker ps -a --filter "name=memlm-" --format "table {{.Names}}\t{{.Status}}"
Get-ChildItem $Out -Directory | ForEach-Object {
    Write-Host "`n== $($_.Name)"
    if (Test-Path "$($_.FullName)\job.log") { Get-Content "$($_.FullName)\job.log" -Tail 3 }
}
