# vi path loss weight
vi_gradient=3
vi_res=2
vi_fusion_gradient=3
vi_fusion_intensity=2

# ir path loss weight
ir_intensity=2
ir_res=3
ir_fusion_intensity=2
ir_fusion_gradient=3

# fusion image loss weight
vi_loss=1
ir_loss=1
fusion_vi_gradient=17
fusion_vi_intensity=10
fusion_ir_gradient=16.001
fusion_ir_intensity=11
dir_name='auto'

# loss weight need to tuning
# fusion_vi_intensity_sets=[24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9]
fusion_vi_intensity_sets=[9+0.1*i for i in range(30)]
# fusion_vi_gradient_sets=[15 + 0.1*i for i in range(30)] 
# fusion_vi_gradient_sets = [16.5]
fusion_vi_gradient_sets=[16.8]



with open('auto.sh', 'w') as f:
    for i in range(len(fusion_vi_gradient_sets)):
        for j in range(len(fusion_vi_intensity_sets)):
            # dir_name will not change 
            dir_name = 'auto_0603_'+str(i*len(fusion_vi_intensity_sets)+j+1)

            # here is the tuning arguments
            fusion_vi_gradient = fusion_vi_gradient_sets[i]
            # fusion_ir_gradient = fusion_vi_gradient 
            fusion_vi_intensity = fusion_vi_intensity_sets[j]
            # fusion_ir_intensity = fusion_vi_intensity

            # write cmd into shell file
            train = 'python main.py \
                    --vi_gradient={} --vi_res={} --vi_fusion_gradient={} --vi_fusion_intensity={} \
                    --ir_intensity={} --ir_res={} --ir_fusion_intensity={} --ir_fusion_gradient={} \
                    --vi_loss={} --ir_loss={} \
                    --fusion_vi_gradient={} --fusion_vi_intensity={} --fusion_ir_gradient={} --fusion_ir_intensity={} \
                    --dir={} \n'.format(vi_gradient, vi_res, vi_fusion_gradient, vi_fusion_intensity,
                        ir_intensity, ir_res, ir_fusion_intensity, ir_fusion_gradient,
                        vi_loss, ir_loss, fusion_vi_gradient, fusion_vi_intensity, fusion_ir_gradient, fusion_ir_intensity,
                        dir_name)
            test = 'python test.py --dir={} --epochs={} \n'.format(dir_name, 10)
            measure = 'python measure.py --dir={} --epochs={} \n'.format(dir_name, 8)
            exp_cmd = train + test  +'\n'
            f.write(exp_cmd)