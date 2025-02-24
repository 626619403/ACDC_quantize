# ACDC_quantize

所有运行代码都在acdc_q的包里，库文件在对应的文件夹。

## TODO

- 在每个数据集上使用不同的模型跑一张大表 这个得小调一下acdc子文件夹下面的数据加载和模型加载
- 所有的运行结果记得保存到wandb上 最后要统计最终边数等
- 做消融实验（删除时间调度前后的耗时）
- 添加attribution patching为baseline
- 

## 注意

- 这个仓库的文件夹下python文件如需使用都需要在对应位置替换，替换的位置在环境的对应位置（我的是miniconda3/lib/python3.12/site-packages/）。其中cmapy文件必须被替换，否则冲突。
- 对应论文是 https://www.overleaf.com/project/6777f3fb672efc7c99aa256c 

## ACDC 原版与改版

- torch>=2.5, cuda>=12.4, triton==3.2 GPU:H800
- requirements里面有是新的依赖文件，不要用原来网站上的poetry。下载完后记得换文件并将triton变回3.2。
- jaxlib和jax要手动装。选0.49版本，0.5没有lib，低版本依赖冲突。
- 若要使用原版:
  - 将解压下的文件中所有带org的装到原版的位置
  - 将experiment中autodl-tmp/acdc_q/acdc/TLACDCExperiment.py中的if self.current_node != None and self.current_node.index.hashable_tuple:及其包含的代码删除\
  - 可以用bash脚本，但记得改项目名称，登录wandb。

## 运行命令
python acdc/main.py \
    --task  docstring\
    --threshold 0.001 \
    --using-wandb \
    --wandb-project-name acdc \
    --wandb-dir "/root/autodl-tmp/wandb" \
    --wandb-mode online \
    --device cuda \
    --reset-network 0 \
    --metric docstring_metric

