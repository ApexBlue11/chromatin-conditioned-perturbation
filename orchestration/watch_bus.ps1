<#
  watch_bus.ps1 — block until a new message appears on a bus lane, then return its path.

  Used by BOTH sessions:
    PI      : powershell -File orchestration\watch_bus.ps1 -Lane to_pi
    Critic  : powershell -File orchestration\watch_bus.ps1 -Lane to_critic

  Exit codes:
    0  NEW      - one or more unprocessed messages exist; paths printed
    2  TIMEOUT  - nothing arrived within -TimeoutSec
    3  STOPPED  - orchestration/STOP exists; both loops must halt

  GOTCHA, learned 2026-09-20. If you background this from a shell that appends anything after it
  (e.g. `; echo done`), the shell returns the LAST command's status and a TIMEOUT becomes
  indistinguishable from a NEW message in the harness notification. Always propagate:

      powershell -File watch_bus.ps1 -Lane to_pi -TimeoutSec 3600; c=$?; echo "EXIT=$c"; exit $c

  Or just read line 1 of the output, which is always NEW / TIMEOUT / STOPPED. Do not infer a wake
  from the notification's exit code alone.

  It does not "wake" an agent by itself — an agent calls it and blocks. The agent-side loop is
  Claude Code's /loop skill, which re-invokes the agent; this script is what makes each wake cheap,
  because it returns instantly when there is nothing to do.
#>
param(
  [Parameter(Mandatory = $true)][ValidateSet('to_pi', 'to_critic')][string]$Lane,
  [int]$TimeoutSec = 300,
  [int]$PollSec = 10
)

$ErrorActionPreference = 'Stop'
$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$stop    = Join-Path $root 'STOP'
$laneDir = Join-Path (Join-Path $root 'bus') $Lane
$seenDir = Join-Path (Join-Path $root 'bus') 'processed'

New-Item -ItemType Directory -Force -Path $laneDir, $seenDir | Out-Null

function Get-Pending {
  Get-ChildItem $laneDir -Filter '*.md' -File -ErrorAction SilentlyContinue |
    Sort-Object Name |
    Where-Object { -not (Test-Path (Join-Path $seenDir $_.Name)) }
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ($true) {
  if (Test-Path $stop) {
    Write-Output "STOPPED"
    Write-Output (Get-Content $stop -Raw -ErrorAction SilentlyContinue)
    exit 3
  }

  $pending = Get-Pending
  if ($pending) {
    Write-Output "NEW"
    $pending | ForEach-Object { Write-Output $_.FullName }
    exit 0
  }

  if ((Get-Date) -ge $deadline) {
    Write-Output "TIMEOUT"
    exit 2
  }
  Start-Sleep -Seconds $PollSec
}
