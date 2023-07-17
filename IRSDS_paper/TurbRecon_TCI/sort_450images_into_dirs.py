import os
import shutil

path_to_input_dir = r'C:\Users\david.j.carr2\OneDrive - US Navy-flankspeed\Documents\JSSAP\IRSDS_paper\TurbRecon_TCI-master\450images'
input_dir = 'images'

print('Starting...')

working_dir = os.path.join(path_to_input_dir, input_dir)
# print(working_dir)
for filename in os.listdir(working_dir):
    f = os.path.join(working_dir, filename)
    file_string, _ = filename.split('.')
    string_array = file_string.split('_')
    fp = string_array[-3]
    i_num = string_array[-2]
    output_dir = os.path.join(path_to_input_dir, fp+'_'+i_num)

    # If directory doesn't exist, make it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    condition = string_array[0]+'_'+string_array[1]+'_'+fp+'_'+i_num
    # print(filename)
    # print(condition)
    if condition in filename:
        shutil.copy(f, output_dir)

print('Done')
