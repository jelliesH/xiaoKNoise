# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:15:31 2016
使用小波分解进行特征提取
分别使用normalize，minmax，scale的方法进行归一化
然后将结果保存到excel中，待matlab训练模型用

@author: 390672
"""
import os, wave, xlrd, wavelet_extract,copy
import numpy as np
import pandas as pd
from time import  time
from pydub import AudioSegment
from prepross_data import Prepross_data
from multiprocessing import Pool, Process, Queue, Manager
"""
读取音频信息和标签表
"""
def read_wav_info_table():
    # 确定目录
    program_path = os.getcwd()
    noise_lib_path = program_path + '/audiolib'
    os.chdir(noise_lib_path)
    file_list = os.listdir()
    os.chdir(program_path)
    
    info_table = dict()
    for file in file_list:
        if '.' in file:
            # 非文件夹的名称            
            pass
        else:
            file_path = noise_lib_path + '/' + file
            info_table_path = file_path + '/' + '导出音频说明文件.xls'
            info_table[file] = [xlrd.open_workbook(info_table_path),file_path]
    return info_table
     
"""
根据读到的信息表获取音频的文件名，机组状态等信息
现阶段暂时只读取机组状态，开放接口，后续补充
返回值：wav_tag_table  音频的信息表(包含各种信息)
"""
def get_wav_info(info_table):
    wav_tag_table = []
    for table_name in info_table:
        book = info_table[table_name][0]
        # 噪音存放的文件夹位置
        file_path  =  info_table[table_name][1]
                
        table= book.sheet_by_name('导出音频属性')
    # machine_status机组状态的数组        
        machine_status_array =np.array( table.col_values(12)[1:])
    #   export_id导出序号
        export_id_array = np.array(table.col_values(0)[1:])
    #   machine_type    机型型号
        machine_type_array = np.array(table.col_values(2)[1:])
    #  wav_name 音频文件名
        wav_name_array = np.array(table.col_values(20)[1:])
    #   wav_type 音频类型
        wav_type_array = np.array(table.col_values(21)[1:])
        
        "建立字典存放音频信息"
        for each_line in export_id_array:
            wav_file_name = '%d_' % each_line + wav_name_array[export_id_array == each_line][0] + \
                                    wav_type_array[export_id_array == each_line][0]
            wav_info = dict()
            wav_info['机组状态'] =  machine_status_array[export_id_array == each_line][0]
            wav_info['导出序号'] = export_id_array[export_id_array == each_line][0]
            wav_info['机型型号'] = machine_type_array[export_id_array == each_line][0]
            wav_info['音频文件名'] = wav_file_name
            wav_info['音频类型'] = wav_type_array[export_id_array == each_line][0]
            wav_info['音频所在地址'] = file_path
            wav_tag_table.append(wav_info)
            
    return wav_tag_table

"""
读取wav文件的内容
返回值：wave_data, framerate    噪音的时间序列，采集数据的帧频率
"""
def get_wave_data(filename,pathname):
    wave_data = np.nan
    framerate = np.nan
    success = False
    if filename.split('.')[-1] in ['mp3','WAV','wav']:
        pass
    else:
        return wave_data, framerate, success
        
    if filename.split('.')[-1] == 'mp3':
        new_filename = filename.replace('mp3','WAV')
        
        file = '/'.join([pathname,filename]) #旧的文件路径
        new_file = '/'.join([pathname,new_filename]) #新的文件路径
        # mp3转换wav
        try:
            sound = AudioSegment.from_mp3(file)
        except FileNotFoundError as e:
            # 音频数据缺失不存在
            print('读取文件 ' + file + ' 的时候出现以下错误：\n ',  e)           
            return wave_data, framerate, success
        sound.export(new_file, format="wav")
        au_file = wave.open(new_file,'rb')
        os.remove(new_file)
    else:
        new_file = '/'.join([pathname,filename]) #新的文件路径
        try:        
            au_file = wave.open(new_file,'rb')
        except BaseException as e:
            # 音频数据缺失不存在
            print('读取文件 ' + new_file + ' 的时候出现以下错误：\n ',  e)            
            return  wave_data, framerate, success 
    
    params = au_file.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    au_data = au_file.readframes(nframes)
    au_file.close()
    wave_data = np.fromstring(au_data, dtype=np.short)
 
    if nchannels == 1:
        pass
    else:
        # 双声道合并成单声道
        # 如果长度是奇数会报错，故此处要处理
        if (len(wave_data)%2 == 1):
            wave_data = np.delete(wave_data,-1)
        else:
            pass
        wave_data = wave_data.reshape(-1, 2)
        wave_data = wave_data.T
        wave_data = wave_data[0] + wave_data[1]
    success = True
    return wave_data, framerate, success

"""
音频剪切，每一段保持10s左右
"""
def audio_cut(wave_data,framerate):
    time_axis = np.arange(0, len(wave_data)) * (1.0 / framerate)
    time_series = pd.Series(time_axis)
    wave_series = pd.Series(wave_data)
    wave_df = pd.DataFrame(columns = ['time_series','wave_series'])
    wave_df['time_series'] = time_series
    wave_df['wave_series'] = wave_series
    
    quo,mod = divmod(time_series.max(),10)
    wave_segment = list()
    
    for ii in range(int(quo)+1):
        if (ii+1)*10 <= time_series.max():
            # 找出小于第ii个10s的音频数据
            wave_segment.append(wave_df[wave_df['time_series'] < (ii+1)*10 ])
            # 将剩余的大于等于第ii个10s的音频数据留给下次判定
            wave_df = wave_df[wave_df['time_series'] >= (ii+1)*10 ]
        else:
            if mod >= 7:
                wave_segment.append(wave_df)
    return wave_segment
    

def del_errpoint_chird(value, input_series):
    ind = input_series[input_series['values'] == value].index
    outcome = [value,input_series.ix[ind + 1].values[0][0]]
    return outcome
    
"""
剔除信号的野点
"""
def del_errpoint(wave_data):
    #   信号的均值
    wave_aver = np.average(wave_data)
    #   信号的标准差
    wave_std = np.std(wave_data)
    wave_Ser = pd.Series(wave_data)
    #   获取信号每个点的索引
    wave_Ser_ind = wave_Ser.index
    #   将原信号的序列中的野点置成nan
    wave_Ser[(wave_Ser > (wave_aver + 3*(wave_std)))\
               |(wave_Ser < (wave_aver - 3*(wave_std)))] = np.nan

    # 野点的索引序列
    mask = wave_Ser.isnull()
    err_point_ind = wave_Ser_ind[mask]
    
    #   normal_point_ind是非野点的索引序列
    mask = wave_Ser.notnull()
    normal_point_ind = wave_Ser_ind[mask]
    normal_point_ind_se = normal_point_ind.to_series()
    
    # normal_point_ind_del: 将normal_point_ind_ser最后一个删除,以和diff之后的结果对称
    normal_point_ind_del = normal_point_ind_se.drop(normal_point_ind_se.index[-1])
    
    # normal_near_err_front_ind： 对normal_point_ind求差，差不为1的就是野点前的一个点
    nor_diff = np.diff(normal_point_ind_se)
    normal_near_err_front_ind = normal_point_ind_del[nor_diff != 1]
    
    # normal_point_ind_ser_re： index重排的normal_point_ind_ser
    normal_point_ind_ser_re = pd.DataFrame(normal_point_ind.values,\
                                       index=np.arange(len(normal_point_ind)),\
                                       columns=['values'])
    # pair_list：包含野点的前一个点和后一个点                          
    pair_list = [del_errpoint_chird(each,normal_point_ind_ser_re)\
                    for each in normal_near_err_front_ind]
    # 用野点的前一个点和后一个点的均值替代野点                  
    for each in pair_list:
        wave_Ser[each[0]+1:each[1]] = np.mean([ wave_Ser[each[0]], wave_Ser[each[1]] ])
    
    # 下面处理如果第一个或最后一个是野点的情况             
    if 0 in err_point_ind:
        # 如果第一点是野点，则第一个非野点至第一个野点之间都是用最后一个非野点的值来替代
        wave_Ser[0:normal_point_ind[0]] = wave_Ser[normal_point_ind[0]]
    else:
        pass
    
    if len(wave_Ser)-1 in err_point_ind:
        # 如果最后一点是野点，则最后一个非野点至最后一个野点之间都是用最后一个非野点的值来替代
        wave_Ser[normal_point_ind[-1]:len(wave_Ser)] =\
                                            wave_Ser[normal_point_ind[-1]]
    else:
        pass
    
    return wave_Ser.values
   
"""
特征提取
"""    
def get_feature(wav_tag_table,var_dict,name):
    print('正在运行第 %s个线程任务 (%s)...' % (name, os.getpid()))    
    wave_mode = var_dict['wave_mode']   
    feature_type = var_dict['feature_type']    
    
    ind = 1
    for wav_info in wav_tag_table:
        print('正在运行第%s个线程任务 (%s)的第%d个文件读取 ' % (name, os.getpid(),ind))
        t_a = time()
        ind = ind+1
        ###   读音频文件
        (wave_data, framerate, success) = get_wave_data( wav_info['音频文件名'] ,wav_info['音频所在地址'] )
        if  success :
            pass
        else:
            continue
        
        ###   构建特征向量的列名以及存放特征向量的字典
        #     在读第一个音频的时候使用
        if ind == 1:
            w_obj = wavelet_extract.Wavelet_analyze(wave_data,framerate)
            # 此处选择小波分析的模式            
            exec('w_obj.' + wave_mode + '(wavelet, analyze_level)')
            # 此处获取小波的特征向量的列名
            col_name = w_obj.col_name
            feature_dict = dict()
            for each in col_name:
                feature_dict[each] = list()
            feature_dict['噪音类型'] = list()
        else:
            pass
        
 
        ###   对每个录音进行剪切，保持大于等于7s
        wave_segment = audio_cut(wave_data,framerate)
        
        for each in wave_segment:
            
            wave_data = each['wave_series'].values
    #########   进行音频能量特征向量的提取 ### start-1 #######
            ### 去除信号的野点,同时做预处理
            wave_data = del_errpoint(wave_data)
            wave_data_pp = Prepross_data(wave_data)
            wave_data_pp = wave_data_pp.normalize_data()
       
            w_obj = wavelet_extract.Wavelet_analyze(wave_data_pp,framerate)
            #   进行小波分解            
            exec('w_obj.' + wave_mode + '(wavelet, analyze_level)')
            #   获取音频的特征向量
            if feature_type == 'pdv':
                feature = w_obj.power_distance()
            else:
                feature = w_obj.wprvector()
            
            for ii in np.arange(len(col_name)):
                feature_dict[col_name[ii]].append(feature[ii])
            feature_dict['噪音类型'].append(wav_info['机组状态'])
            print('完成第%s个线程任务 (%s)的第%d个文件读取！' % (name, os.getpid(),ind))
    #########   进行多种归一化实验尝试 ### end-1 #########        
            
        t_b = time()
        print('第 %s个线程任务 (%s)已完成,共耗时：%f秒\n' % (name, os.getpid(),(t_b - t_a)))
        return feature_dict
"""
多线程处理特征提取的任务
wav_tag_table 完整的噪音信息表
n_process 多线程的线程数
q 存放每个线程运行结果的队列
"""
def multi_process(wav_tag_table,n_process,q,var_dict):
    ### 对数据集进行分派，划分成n_process份，n_process为线程数：
    dev = len(wav_tag_table)//n_process
    rem = len(wav_tag_table)%n_process
    
    if rem != 0:
        num_total = (dev+1)*n_process
        num_range = np.arange(num_total)
        num_me = num_range.reshape((n_process,-1))
    else:
        num_range = np.arange(len(wav_tag_table))
        num_me = num_range.reshape((n_process,-1))
    # 索引对
    ind_pair = [[each[0],each[-1]+1] for each in num_me]    
    # 根据索引对将总的数据集划分成多个数据集
    wav_tag_table_package = [wav_tag_table[each[0]:each[1]] \
                    for each in ind_pair]
    ### 创建进程池
    print('Process(ID: %s) begin'% os.getpid())
    p = Pool(n_process)
    for ii in np.arange(n_process):
        ### 对进程池中的每个进程分配get_feature任务
        result = p.apply_async(get_feature, args=(wav_tag_table_package[ii],\
                var_dict,ii))
        q.put(result)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

"""
主程序入口
"""
if __name__ == '__main__':
    t_start = time()
###   1. 获取音频信息表
    info_table = read_wav_info_table()
###   2. 获取各个音频的信息和标签
    wav_tag_table =  get_wav_info(info_table)
    ind = 0
    total_num  = len(wav_tag_table) 
###   3. 构建特征向量的列名以及存放特征向量的字典
    feature_dict = dict()
    feature_dict['噪音类型'] = list()
    
###   4. 对每个录音进行综合特征提取
###   先确定该次分解的相关设置
    # 小波分析模式 'wavelet_tree'，'wavelet_dec'
    # wavelet 选用的小波函数
    # analyze_level 分解层数
    # 使用 能量距特征'pdv' 还是 能量特征 'pv'
    wave_mode = 'wavelet_tree'    
    wavelet = 'db5'
    analyze_level = 5
    feature_type = 'pdv'
    # 构成参数字典，后续调用的程序get_feature需要用到
    var_dict = {'wave_mode':wave_mode,
                'wavelet':wavelet,
                'analyze_level':analyze_level,
                'feature_type':feature_type} 
    # 启动多线程执行特征提取的任务
    # 存放线程之间执行结果的队列
     
    q = Queue()
    # 需要启动的线程数
    n_process = 7
    pw = Process(target=multi_process, args=(wav_tag_table,n_process,q,var_dict,))
    pw.start()
    pw.join()
    file_prefix = wave_mode + '_' + wavelet + '_' + \
                        str(analyze_level) + '_' + feature_type    
#    # 将结果转换成易于处理的DataFrame
#    data_source = pd.DataFrame(feature_dict)
#    
#    # 将结果保存成excel格式，下次直接读入excel
#    wbw = pd.ExcelWriter(file_prefix + 'data_source.xlsx')
#    data_source.to_excel(wbw,'data_source')
#    wbw.save()
#    print('程序总耗时：%f 秒'%(time() - t_start))