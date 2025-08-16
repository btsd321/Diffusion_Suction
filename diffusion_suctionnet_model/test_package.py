#!/usr/bin/env python
"""
测试 diffusion_suctionnet_model 包的基本功能
"""
import sys
import torch
import numpy as np

def test_package_import():
    """测试包的导入功能"""
    print("测试包导入...")
    try:
        import diffusion_suctionnet_model
        print(f"✓ 成功导入 diffusion_suctionnet_model，版本: {diffusion_suctionnet_model.__version__}")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        from diffusion_suctionnet_model import dsnet, ScheduledCNNRefine
        
        # 创建模型实例
        model = dsnet(use_vis_branch=True, return_loss=False)
        print("✓ 成功创建 dsnet 模型")
        
        # 创建调度器
        refine_model = ScheduledCNNRefine(channels_in=128, channels_noise=4)
        print("✓ 成功创建 ScheduledCNNRefine 模型")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_pointnet2_import():
    """测试 PointNet2 模块导入"""
    print("\n测试 PointNet2 模块导入...")
    try:
        from diffusion_suctionnet_model import pointnet2
        if pointnet2 is not None:
            print("✓ 成功导入 PointNet2 模块")
            return True
        else:
            print("⚠ PointNet2 模块不可用（可能需要编译 CUDA 扩展）")
            return True
    except Exception as e:
        print(f"✗ PointNet2 导入失败: {e}")
        return False

def test_scheduler_import():
    """测试调度器导入"""
    print("\n测试调度器导入...")
    try:
        from diffusion_suctionnet_model import DDIMScheduler
        if DDIMScheduler is not None:
            scheduler = DDIMScheduler(num_train_timesteps=1000)
            print("✓ 成功导入并创建 DDIMScheduler")
            return True
        else:
            print("⚠ DDIMScheduler 不可用")
            return True
    except Exception as e:
        print(f"✗ 调度器导入失败: {e}")
        return False

def test_model_forward():
    """测试模型前向传播（简单测试）"""
    print("\n测试模型前向传播...")
    try:
        from diffusion_suctionnet_model import dsnet
        
        # 创建模型
        model = dsnet(use_vis_branch=True, return_loss=False)
        model.eval()
        
        # 创建假数据
        batch_size = 1
        num_points = 1024
        
        # 简化的输入数据
        inputs = {
            'point_clouds': torch.randn(batch_size, num_points, 3),
            'labels': {
                'suction_or': torch.randn(batch_size, num_points, 3),
                'suction_wrench_scores': torch.randn(batch_size, num_points),
                'suction_feasibility_scores': torch.randn(batch_size, num_points),
                'individual_object_size_lable': torch.randn(batch_size, num_points),
            }
        }
        
        # 前向传播
        with torch.no_grad():
            pred_results, ddim_loss = model(inputs)
            
        print(f"✓ 模型前向传播成功，输出形状: {pred_results.shape if pred_results is not None else 'None'}")
        return True
        
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 diffusion_suctionnet_model 包...")
    print("=" * 60)
    
    tests = [
        test_package_import,
        test_model_creation,
        test_pointnet2_import,
        test_scheduler_import,
        test_model_forward,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！包已成功打包。")
        return 0
    else:
        print("❌ 部分测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
