with open('eval.sh','w') as f:
    for i in range(30):
        dir_name = 'auto_0603_' + str(i+1)
        cmd = 'python measure.py --dir={} --epochs={} & \n'.format(dir_name, 10)
        f.write(cmd) 