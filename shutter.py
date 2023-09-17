import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

for i in range(1,3701):
    dataname='hp/hp_'+str(i)+'.npy'
    timename='hp/hp_time_'+str(i)+'.npy'
    paraname='hp/hp_para_'+str(i)+'.npy'

    noisename = 'ts/ts_' + str(i) + '.npy'
    noisetime = 'ts/ts_time_' + str(i) + '.npy'

    hp_sample_times=np.load(timename)
    hp = np.load(dataname)
    hp_para=np.load(paraname)
    ts_sample_times = np.load(noisetime)
    ts = np.load(noisename)

    ts=100*ts

    old_value = i
    old_min = 1
    old_max = 3700
    new_min = 1
    new_max = 200

    new_value = (old_value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    # print(new_value)
    t=0.01*new_value  #0.01-2
    # print(t)
    xx=int(t/0.0002)

    hp_sample_times = np.append(hp_sample_times[:xx], hp_sample_times + t)
    hp_fake=np.array([0]*xx)
    hp_fake=np.append(hp_fake,hp)
    hp=np.append(hp,[0]*xx)

    hp=hp+hp_fake
    # 创建数据框
    # plt.plot(ts_sample_times, ts)
    # plt.plot(hp_sample_times+10, hp,color='orange', label=hp_para[0])

    # hp_para = [float(i) for i in hp_para[1:]]
    #
    # plt.title('m1: {:.1f} m2: {:.1f} inclination: {:.3f} coa_phase: {:.3f} distance: {:.2f}'
    #          .format(hp_para[0], hp_para[1], hp_para[3], hp_para[4], hp_para[6]))
    # plt.ylabel('Strain')
    # plt.xlabel('Time (s)')
    # plt.legend()
    # plt.show()
##################################################################
    target_length = 75000
    # 计算需要填充的长度
    padding_length = target_length - len(hp)# 计算左侧和右侧需要填充的长度
    left_padding = padding_length // 2
    right_padding = padding_length - left_padding
    # 执行填充
    hp = np.pad(hp, (left_padding, right_padding), mode='constant')
    #print(hp)
    mix=hp+ts
    # plt.plot(ts_sample_times,mix)
    # plt.show()
    # 示例引力波时域数据
    time_domain_data =mix # 在这里插入pycbc生成的引力波时域数据

    # STFT参数
    window_size = 1024
    hop_size = 256

    # 计算STFT
    frequencies, times, spectrogram = signal.stft(time_domain_data, window='hann', nperseg=window_size, noverlap=window_size-hop_size)

    # 绘制STFT图像
    # plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spectrogram), aspect='auto', cmap='jet', origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.title('STFT Spectrogram')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show()
    plt.axis('off')
    img_path = "./combined"
    img_name = 'cbd_'+str(i)+'.png'
    path_img_name = os.path.join(img_path, img_name)
    plt.savefig(path_img_name,bbox_inches='tight', pad_inches=0,dpi=300)
    plt.close()
    print(i)