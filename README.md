I have created dynamic code that allows you to control the depth of the block. I named it 'layer' as I prefer the name. You can set any depth you desire. I observed that there are some architectures with a depth of 5 blocks inside an extra block, which is more general, in my opinion.

I have written the code with only one line, and you can compare multiple models with the same or different models. You can also use either the kernel or linear CKA. Furthermore, there are other parameters in the code that control the input, such as data, batch size, data loader size, pre-trained models, conv_only, etc.

I display for each pair of models the heatmap and save the information about the command.
