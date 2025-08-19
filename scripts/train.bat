@echo off
REM ====================================================================
REM Diffusion_Suction 训练脚本启动器
REM ====================================================================
REM 功能: 启动 Diffusion_Suction 项目的 train.py 脚本
REM 位置: 本脚本应放在 scripts 文件夹下，使用相对路径启动主目录下的 train.py
REM 作者: btsd321
REM 日期: 2025年8月17日
REM ====================================================================

REM 切换到项目根目录（scripts的上级目录）
cd /d %~dp0..

REM 激活 conda 环境（环境名: window_conda）
REM 如需修改环境名，请更改下行中的 window_conda
call D:/SoftWare/anaconda3/Scripts/activate.bat window_conda

REM ====================================================================
REM 训练参数配置
REM ====================================================================
python train.py ^
  --data_dir "G:/Diffusion_Suction_DataSet/train" ^
  --train_cycle_list "[0,69]" ^
  --train_scene_list "[1,50]" ^
  --test_cycle_list "[70,89]" ^
  --test_scene_list "[1,50]" ^
  --output_dir "D:/Project/Diffusion_Suction/output" ^
  --device_list "0" ^
  --batch_size 8 ^
  --base_learning_rate 0.01 ^
  --max_epoch 200

@REM python train.py ^
@REM   --data_dir "G:/Diffusion_Suction_DataSet/train" ^
@REM   --train_cycle_list "[0,69]" ^
@REM   --train_scene_list "[1,50]" ^
@REM   --test_cycle_list "[70,89]" ^
@REM   --test_scene_list "[1,50]" ^
@REM   --output_dir "D:/Project/Diffusion_Suction/output" ^
@REM   --device_list "0" ^
@REM   --batch_size 16 ^
@REM   --base_learning_rate 0.01 ^
@REM   --max_epoch 500

pause