param(
    [ValidateRange(1, 120)]
    [int]$Frames = 120,
    [ValidateRange(0.01, 10.0)]
    [double]$Delay = 0.35,
    [switch]$Fullscreen
)

$fullscreenArg = if ($Fullscreen) { " --fullscreen" } else { "" }
$linuxCommand = @"
source /home/hzy/miniconda3/etc/profile.d/conda.sh &&
conda activate opencood &&
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/hzy/miniconda3/envs/opencood/lib &&
cd /mnt/d/shixi/Coop3d-system &&
python tools/infer_single_opv2v.py \
  --opencood-root /mnt/d/shixi/OpenCOOD \
  --model-dir /mnt/d/shixi/checkpoints/pointpillar_late_fusion \
  --data-root /mnt/d/shixi/datasets/OPV2V-live/test \
  --output-dir /mnt/d/shixi/Coop3d-system/outputs/live_demo \
  --max-frames $Frames \
  --live --live-delay $Delay$fullscreenArg
"@

Write-Host "Starting real inference for $Frames frames..." -ForegroundColor Cyan
Write-Host "Close the BEV window after the final frame to exit." -ForegroundColor DarkGray
wsl.exe -d Ubuntu-22.04 -- bash -lc $linuxCommand
