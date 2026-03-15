# Horizon Ledger — Windows Task Scheduler Setup
# Run once from an elevated (Admin) PowerShell:
#   cd D:\Stock_Analysis\horizon-ledger\scripts
#   .\setup_scheduler.ps1

$python = "D:\Stock_Analysis\horizon-ledger\.venv\Scripts\python.exe"
$scriptsDir = "D:\Stock_Analysis\horizon-ledger\scripts"

# ── Daily Update (Mon–Fri at 8:00 PM) ────────────────────────────────────────
$dailyAction = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$scriptsDir\run_daily.py`"" `
    -WorkingDirectory "D:\Stock_Analysis\horizon-ledger"

$dailyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "20:00"

$dailySettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "HorizonLedger\DailyUpdate" `
    -Action $dailyAction `
    -Trigger $dailyTrigger `
    -Settings $dailySettings `
    -Description "Horizon Ledger — fetch prices, score stocks, update market digest (weekdays 8 PM)" `
    -RunLevel Highest `
    -Force

Write-Host "✅ DailyUpdate task created (Mon–Fri 8:00 PM)" -ForegroundColor Green

# ── Weekly Update (Saturday at 7:00 AM) ──────────────────────────────────────
$weeklyAction = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$scriptsDir\run_weekly.py`"" `
    -WorkingDirectory "D:\Stock_Analysis\horizon-ledger"

$weeklyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Saturday `
    -At "07:00"

$weeklySettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "HorizonLedger\WeeklyUpdate" `
    -Action $weeklyAction `
    -Trigger $weeklyTrigger `
    -Settings $weeklySettings `
    -Description "Horizon Ledger — weekly scoring, reconstitution proposals, portfolio sync (Sat 7 AM)" `
    -RunLevel Highest `
    -Force

Write-Host "✅ WeeklyUpdate task created (Saturday 7:00 AM)" -ForegroundColor Green

# ── Verify ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Registered tasks:" -ForegroundColor Cyan
Get-ScheduledTask -TaskPath "\HorizonLedger\" | Format-Table TaskName, State, @{
    Name="Next Run"; Expression={ (Get-ScheduledTaskInfo $_.TaskName -TaskPath $_.TaskPath).NextRunTime }
}
