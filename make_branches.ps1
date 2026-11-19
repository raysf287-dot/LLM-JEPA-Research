# Run once from inside this folder:  powershell -ExecutionPolicy Bypass -File .\make_branches.ps1
$ErrorActionPreference = "Stop"
if (Test-Path .git) { throw "Already a git repo. Delete .git first or run the commands by hand." }
git init -b main
git add README.md .gitignore make_branches.ps1
git commit -m "Initial commit: README and gitignore"

$stages = @(
  @{ branch = "stage/1-pilot-latent-predictor";        paths = @("pilot");                    msg = "Pilot P1-P4: dedicated latent predictor on frozen backbone" },
  @{ branch = "stage/2-llm-jepa-reproduction";         paths = @("src","reproduction");       msg = "Reproduce LLM-JEPA with finetune_step.py and stage 1 runs" },
  @{ branch = "stage/3-controls-and-pipeline-fixes";   paths = @("controls_and_fixes");       msg = "Controls, end-token and annotation fixes, format/epoch tests" }
)
foreach ($s in $stages) {
  git checkout -b $s.branch main
  git add $s.paths
  git commit -m $s.msg
  git checkout main
  git merge --no-ff $s.branch -m ("Merge " + $s.branch)
}
git log --oneline --graph --all
Write-Host "`nDone. To combine with your personal repo later:  git remote add origin <your-repo-url>; git push -u origin --all"
