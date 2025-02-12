# ACDC_quantize

所有代码已经写完并且调好。可以跑通并且结果还行。可以跑数据了。

## 注意

- 所有在这个仓库的python文件如需使用都需要在对应位置替换，替换的位置在文件的开头会写。其中cmapy文件必须被替换，否则冲突。
- ACDC的代码还未彻底改完。如果想要跑，可以看最下面的安装和bug修正。由于ACDC代码过多，放上来很乱，我先不放了。
- attention文件还未彻底完成，这个版本仅供检查量化和理论方面是否有误。***王良宇*** 良宇哥可以看一下这个量化方法有没有什么问题。
- 对应论文是 https://www.overleaf.com/project/6777f3fb672efc7c99aa256c 。目前只写了引言。***杨澍*** 树姐可以改一下。
- wandb仓库是 https://wandb.ai/rat-nest/acdc/overview 。

## TODO

- 得把ACDC的subnetwork_probing和launch_all_sixteen_heads跑一下。***张林*** 林哥有时间可以跑一下上传到wandb上。脚本用16heads的那个可以当模板。
- 量化的todo写在attention文件最顶上了。明天改完全部的代码开始跑实验。
- 目前的文章的参考文献和理论部分可以开始写了。我明天得改代码，所以只能抽出比较少的时间。
- naive的我已经跑完了。需要跑原版的代码以获得数据。
- 如果使用仓库里的脚本记得把wandb任务名字改成original-xxx，建议把线程数调成1并换小卡跑，否则会造成edge文件冲突。改后的得等我把代码写完。
- tracr任务 代码离谱bug多，得换个任务或者去掉。

## ACDC bug和安装

- ACDC bug有点多，安装比较麻烦，建议等我明天把代码全部改完再搞。现在这个改了大部分的bug，但没有做量化的适配。
- 先参照ACDC的把要用apt的安好，然后python一定要选3.9以上，torch必须选2.0以上，否则会出list的选取bug。
- requirements里面有是新的依赖文件，不要用原来网站上的poetry。
- 如果你非要用官网的poerty，改掉下面tracr的仓库地址后，把锁重新加载后要自己手动补很多的包。而且它文件里就是冲突的，有的包版本根本找不到，还得手动配依赖。
- tracr的仓库已经改到了https://github.com/google-deepmind/tracr而不是原来的https://github.com/deepmind/tracr
- or-gate任务的reset_network模型权重在huggingface上不见了。跑脚本的时候记得把这个去了或者在main.py的298行把条件改成if RESET_NETWORK and TASK!="or_gate":。
- 这个ACDC所有的模型存储位置全都默认是cpu，而且所有的wandb仓库组织和任务名称全部是写死的。
- 每个主代码之间的device参数不互通，最离谱的是有在代码里写死的cpu上的张量，acdc/logic_gates/utils.py get_all_logic_gate_things函数里必须把第三行data = torch.tensor([[0.0]]).long()移到device上，因为如果不手动移这个张量，这个张量生成的correct_answers会在后面的partial里被固定，就会报模型不在一个设备上的错。
- jaxlib和jax要手动装，requirement文件里路径没找到是错的。一定要选0.49版本，0.5没有lib，低版本依赖冲突。
