import os


data_dir = '/home/aistudio/work/UCF-101-jpg/'

train_data = os.listdir(data_dir + 'train01')
train_data = [x for x in train_data if not x.startswith('.')]
print(len(train_data))

test_data = os.listdir(data_dir + 'test01')
test_data = [x for x in test_data if not x.startswith('.')]
print(len(test_data))

# val_data = os.listdir(data_dir + 'val')
# val_data = [x for x in val_data if not x.startswith('.')]
# print(len(val_data))

f = open(data_dir+ 'train01.list', 'w')
for line in train_data:
    f.write(data_dir + 'train/' + line + '\n')
f.close()
f = open(data_dir+ 'test01.list', 'w')
for line in test_data:
    f.write(data_dir + 'test/' + line + '\n')
f.close()
# f = open(data_dir+ 'val.list', 'w')
# for line in val_data:
#     f.write(data_dir + 'val/' + line + '\n')
# f.close()

