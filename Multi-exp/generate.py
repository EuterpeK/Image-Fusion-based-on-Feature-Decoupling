# vi path loss weight
vi_gradient=1.8
vi_res=1.5
vi_fusion_gradient=1.8
vi_fusion_intensity=1.5

# ir path loss weight
ir_intensity=1.5
ir_res=1.8
ir_fusion_intensity=1.5
ir_fusion_gradient=1.8

# fusion image loss weight
vi_loss=1
ir_loss=1
fusion_vi_gradient=18
fusion_vi_intensity=1.5
fusion_ir_gradient=19
fusion_ir_intensity=1.7
dir_name='auto'

# loss weight need to tuning
fusion_vi_intensity_sets=[1.35, 1.37, 1.39, 1.41, 1.43, 1.45, 1.47, 1.49]
fusion_ir_intensity_sets=[1.5]


with open('auto.sh', 'w') as f:
    for i in range(len(fusion_vi_intensity_sets)):
        for j in range(len(fusion_ir_intensity_sets)):
            # dir_name will not change 
            dir_name = 'auto_'+str(i*len(fusion_ir_intensity_sets)+j+1)

            # here is the tuning arguments
            fusion_vi_intensity = fusion_vi_intensity_sets[i]
            fusion_ir_intensity = fusion_ir_intensity_sets[j]

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
            test = 'python test.py --dir={} --epochs={} \n'.format(dir_name, 16)
            measure = 'python measure.py --dir={} --epochs={} \n'.format(dir_name, 16)
            exp_cmd = train + test + measure +'\n'
            f.write(exp_cmd)