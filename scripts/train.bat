@echo off
REM 启动 Diffusion_Suction 项目的 train.py 脚本
REM 本脚本应放在 scripts 文件夹下，使用相对路径启动主目录下的 train.py

cd /d %~dp0..

REM 激活 conda 环境（如有需要请修改环境名）
call ..\..\SoftWare\anaconda3\Scripts\activate.bat window_conda

REM 运行训练脚本
python train.py ^
  --data_dir "G:/Diffusion_Suction_DataSet/train" ^
  --train_cycle_list "[0,9]" ^
  --train_scene_list "[1,50]" ^
  --test_cycle_list "[80,89]" ^
  --test_scene_list "[1,50]" ^
  --output_dir "D:/Project/Diffusion_Suction/output" ^
  --device_list "0" ^
  --log_dir "D:/Project/Diffusion_Suction/output/log" ^
  --batch_size 8 ^
  --base_learning_rate 0.003

pause
