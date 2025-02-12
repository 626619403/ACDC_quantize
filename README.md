# ACDC_quantize

所有代码已经写完并且调好。可以跑通并且结果还行。可以跑数据了。所有运行代码都在acdc_q的包里，库文件在对应的文件夹。


## 注意

- 所有在这个仓库的python文件如需使用都需要在对应位置替换，替换的位置在文件的开头会写。其中cmapy文件必须被替换，否则冲突。
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

