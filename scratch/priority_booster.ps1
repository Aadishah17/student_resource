param([int]$ParentPid = 15968)

Write-Output "Priority booster active for PID $ParentPid"
while ($true) {
    $parent = Get-Process -Id $ParentPid -ErrorAction SilentlyContinue
    if (-not $parent) {
        Write-Output "Process $ParentPid has terminated. Booster exiting."
        break
    }
    if ($parent.PriorityClass -ne 'High') {
        try { $parent.PriorityClass = 'High' } catch {}
    }
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentPid" -ErrorAction SilentlyContinue | ForEach-Object {
        $proc = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        if ($proc -and $proc.PriorityClass -ne 'High') {
            try { $proc.PriorityClass = 'High' } catch {}
        }
    }
    Start-Sleep -Seconds 5
}
