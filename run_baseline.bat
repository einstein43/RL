@echo off
echo ======================================================================
echo RUNNING BASELINE DQN EXPERIMENT
echo ======================================================================
echo.
echo This will take approximately 5-10 minutes.
echo The training will run uninterrupted until completion.
echo.
echo Press Ctrl+C now if you want to cancel, otherwise training will start in 3 seconds...
timeout /t 3 >nul 2>&1

cd "c:\Users\alexv\OneDrive\Documents\HBO-ICT\S7 AI\Data challenge\PF Core programme\RL\RL"
python run_baseline_full.py

echo.
echo ======================================================================
echo TRAINING COMPLETE!
echo ======================================================================
echo Check results in: results/dqn_experiments/baseline_final/
echo Check plots in: plots/baseline_final_results.png
echo.
pause
