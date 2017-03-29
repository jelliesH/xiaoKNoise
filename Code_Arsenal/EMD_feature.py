# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:15:31 2016
使用综合指标技术进行特征提取
分别使用normalize，minmax，scale的方法进行归一化
然后将结果保存到excel中，待matlab训练模型用

@author: 390672
"""
import os, wave, xlrd, comfeature
import numpy as np
import pandas as pd
from time import  time
from pydub import AudioSegment
from multiprocessing import Process, Pool
from prepross_data import Prepross_data
import matlab.engine

"""
读取音频信息和标签表
"""
def read_wav_info_table():
    # 确定目录
    program_path = os.getcwd()
    noise_lib_path = '/home/390672/WORKSHOP/noise_analysis/audiolib'
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
主程序入口
"""
if __name__ == '__main__':
    t_start = time()
    #   1. 获取音频信息表
    info_table = read_wav_info_table()
    #   2. 获取各个音频的信息和标签
    wav_tag_table =  get_wav_info(info_table)
    ind = 0
    total_num  = len(wav_tag_table) 
    #   3. 构建24个特征向量的列名以及存放综合特征的字典
    column_name = []
    feature_dict = dict()
    eng = matlab.engine.start_matlab()
    result = list()
#####   4. 对每个录音进行综合特征提取
    for wav_info in wav_tag_table:
        t_a = time()
        ind = ind + 1
        if ind <= 3:
            pass   
        else:
            break
        print('共有%d'%total_num + '个音频，正在读取第%d'%ind + '个')
        #   读音频文件
        (wave_data, framerate, success) = get_wave_data( wav_info['音频文件名'] ,wav_info['音频所在地址'] )
        if  success :
            pass
        else:
            continue
#####   对每个录音进行剪切，保持大于等于7s
        wave_segment = audio_cut(wave_data,framerate)
        
        for each in wave_segment:
            
            wave_data = each['wave_series'].values
    #########   开始对信号进行特征集的提取

            file_prefix = 'normal_normal_'
            
            ##################################
            ##          去除信号野点          ##
            ##################################            
            wave_data = del_errpoint(wave_data)
            #wave_data_pp = Prepross_data(wave_data.reshape(-1,1))
            wave_data_pp = Prepross_data(wave_data)
            wave_data_pp = wave_data_pp.normalize_data()

            #########################################################
            ##          获取 特征集1 和 特征集2 的24个综合特征          ##
            #########################################################
            # f_cl1 - dict()类型
            #   key值是特征的名称，value为具体的特征值
            # f_cl2 同 f_cl1            
            # 单线程 方式版       
            feature = comfeature.Feature(wave_data_pp,framerate)
            f_cl1 = feature.get_time_feature()
            f_cl2 = feature.get_freq_feature()
            #   多线程
            #feature = p.apply_async(comfeature.Feature, args = (wave_data,framerate,))
            print('特征集1与特征集2特征提取完毕！')
            #########################################################
            ##          获取 特征集3 和 特征集4 的综合特征              ##
            #########################################################  
            #   滤波器设计
            #   设计思路：用3个滤波器将信号拆解。分别是：低通/高通/和带通
            #   滤波器数值设置
            #       低通 - 小于500HZ； 带通 - 500～1k，1k~2k,2k~1w； 高通 - 大于1w 
            freq_setting = {'Lowpass':(500,600),
                            'Bandpass':[([500,1000],[400,1100]),
                            ([1000,2000],[900,2100]),([2000,10000],[1900,10100])],
                            'Highpass':(10000,10100)}
            #  数据类型说明
            #  f_cl3s - dict()类型
            #       key值是对应的滤波器的名称，value为该滤波器对应的时域特征集 f_cl3 
            #            f_cl3 - dict()
            #               key值是特征的名称，value为具体的特征值
            #  f_cl4s同f_cl3s
            f_cl3s = dict() 
            f_cl4s = dict()               
            for each in freq_setting:
                mode = each;subname_cl3 = 'CL3_' + each;
                subname_cl4 = 'CL4_' + each;
                if mode is 'Lowpass':
                    fp = freq_setting[each][0]
                    fs = freq_setting[each][1]
                    
                    # 获取特征集 3
                    feature = comfeature.Feature_Cl3(wave_data_pp,framerate,subname_cl3,fs=fs,fp=fp)
                    feature.filter_butter(mode)
                    f_cl3 = feature.get_cl3_feature()
                    f_cl3s[subname_cl3] = f_cl3
                    filtered_wave = feature.filtered_wave # 滤波后的结果
                    
                    # 获取特征集 4
                    feature = comfeature.Feature_Cl4(filtered_wave,framerate,subname_cl4)
                    f_cl4 = feature.hilbert_envelop()
                    f_cl4s[subname_cl4] = feature.get_cl4_feature()
                    
                    
                elif mode is 'Highpass':
                    fp = freq_setting[each][1]
                    fs = freq_setting[each][0]
                    
                    # 获取特征集 3
                    feature = comfeature.Feature_Cl3(wave_data_pp,framerate,subname_cl3,fs=fs,fp=fp)
                    feature.filter_butter(mode)
                    f_cl3 = feature.get_cl3_feature()
                    f_cl3s[subname_cl3] = f_cl3
                    filtered_wave = feature.filtered_wave # 滤波后的结果
                    
                    # 获取特征集 4
                    feature = comfeature.Feature_Cl4(filtered_wave,framerate,subname_cl4)
                    f_cl4 = feature.hilbert_envelop()
                    f_cl4s[subname_cl4] = feature.get_cl4_feature()
                    
                else:
                    # 这是bandpass滤波器
                    N = len(freq_setting[each])
                    subname3 = subname_cl3
                    subname4 = subname_cl4
                    for ii in np.arange(N):
                        subname_cl3 = subname3 + '_' + str(ii)
                        subname_cl4 = subname4 + '_' + str(ii)
                        fp = freq_setting[each][ii][0]
                        fs = freq_setting[each][ii][1]
                        
                        # 获取特征集 3
                        feature = comfeature.Feature_Cl3(wave_data_pp,framerate,subname_cl3,fs=fs,fp=fp)
                        feature.filter_butter(mode)
                        f_cl3 = feature.get_cl3_feature()
                        f_cl3s[subname_cl3] = f_cl3
                        filtered_wave = feature.filtered_wave # 滤波后的结果
                        
                        # 获取特征集 4
                        feature = comfeature.Feature_Cl4(filtered_wave,framerate,subname_cl4)
                        f_cl4 = feature.hilbert_envelop()
                        f_cl4s[subname_cl4] = feature.get_cl4_feature()
            print('特征集3与特征集4特征提取完毕！')            
            #########################################################
            ##          获取 特征集5 和 特征集6 的综合特征              ##
            #########################################################
            # 开启matlab的API，进行EMD
            # 获取特征集5
            
            #  数据类型说明
            #  f_cl5s - dict()类型
            #       key值是对应的imf的名称，value为该imf对应的时域特征集 f_cl5 
            #            f_cl5 - dict()
            #               key值是特征的名称，value为具体的特征值
            #  f_cl6s同f_cl3s
            
            print('正在提取特征集5......')
            subname_cl5 = 'CL5_'
            feature = comfeature.Feature_Cl5(wave_data_pp,framerate,subname_cl5)
            feature.emd_analysis(eng,matlab)
            f_cl5s,imfs = feature.get_cl5_feature()
            
            # 获取特征集6
            subname_cl6 = 'CL6_imf'
            feature = comfeature.Feature_Cl6(imfs,framerate,subname_cl6)
            f_cl6s = feature.get_cl6_feature()
            print('特征集5与特征集6特征提取完毕！')
            
            ############################################################
            ######### 如果是首次提取特征集，需要根据所有特征集的特征   #########
            #########    名字，建立存放所有特征值数据的Dataframe    ######### 
            ############################################################
            c = (ind,f_cl1,f_cl2,f_cl3s,f_cl4s,f_cl5s,f_cl6s)
            result.append(c)
        t_b = time()
        print('读取完成,共耗时：%f'%(t_b - t_a) + '秒')
    eng.quit() 
        
#    # 将结果转换成易于处理的DataFrame
#    data_source = pd.DataFrame(feature_dict)
#    # 将每列特征向量做归一化
#    ds_col_name = data_source.columns
#    ds_col_name = ds_col_name.drop('噪音类型')
#    for each in ds_col_name:
#        each_feature = data_source.loc[:,each].values
#        each_feature_pp = Prepross_data(each_feature)
#        each_feature_pp = each_feature_pp.normalize_data()
#        data_source.loc[:,each] = each_feature_pp
#    
#    # 将结果保存成excel格式，下次直接读入excel
#    wbw = pd.ExcelWriter(file_prefix + 'data_source.xlsx')
#    data_source.to_excel(wbw,'data_source')
#    wbw.save()
    print('程序总耗时：%f 秒'%(time() - t_start))