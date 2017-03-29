# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:56:33 2016
该程序主要用于从本地目录noise_analysis
上读取噪音音频数据，并将他们进行MFCC
将特征提取出来，存放到一个xls文件中
@author: 390672
"""
import os;import wave;import pylab as pl
import numpy as np;import scipy.signal as signal ;from pydub import AudioSegment
# 梅尔倒谱的系数提取需要的关联库
from features import mfcc;from features import logfbank
from features import fbank;from features import ssc
import pandas
import scipy.io.wavfile as wav; import xlwt; import xlrd
from time import  time
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
获取梅尔倒谱特征
返回值：feature_array 梅尔倒谱特征
"""
def get_MFCC_feature(sig,rate):
    p_array = mfcc(sig,rate,winlen=0.025, winstep=0.01) # 获取梅尔倒谱系数
    col_num = p_array.shape[1]
    feature_array = []
    for ii in range(col_num):
        #  test1.  取某一维的标准差
        # feature_array.append(np.std(p_array[:,ii]))
        feature_array.append(p_array[:,ii])
    return feature_array
"""
对数据做归一化
y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
"""
def normalization(x_array,y_min,y_max):
    x_max = np.max(x_array)
    x_min = np.min(x_array)
    y = list(map(lambda x: (y_max - y_min)*(x - x_min)/( x_max - x_min ) + y_min, x_array))
    return y
"""
对输入的数据每一列都做实验性的统计特征提取
"""    
def sta_feature( MFCC_feature ):
    col_num = len(MFCC_feature)
    output_array = []
    for ii in range(col_num):
        # TEST1. 取标准差
        #output_array.append( MFCC_feature[ii].std() )
        # TEST2. 取峰度
        # 峰度是描述总体中所有取值分布形态陡缓程度的统计量。和正态分布曲线做比较
        #output_array.append( MFCC_feature[ii].kurt() )
        # TEST3. 取偏度
        # 偏度是总体取值分布的对称性
        output_array.append( MFCC_feature[ii].skew() )
    return output_array
           
"""
main( )函数
"""    
if __name__ == '__main__':
    t_start = time()
    #   1. 获取音频信息表
    info_table = read_wav_info_table()
    #   2. 获取各个音频的信息和标签
    wav_tag_table =  get_wav_info(info_table) 
    ind = 0
    feature_list = list()
    tag_list = list()
    column_list = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','机组状态']
    #   3. 提取各个音频的特征，并将属性和类型组成数据集
    for wav_info in wav_tag_table:
        ind = ind + 1
        total_num  = len(wav_tag_table)        
        print('共有%d'%total_num + '个音频，正在读取第%d'%ind + '个')
        #   读音频文件
        (wave_data, framerate, success) = get_wave_data( wav_info['音频文件名'] ,wav_info['音频所在地址'] )
        if  success :
            pass
        else:
            continue
        MFCC_feature  = get_MFCC_feature(wave_data, framerate)
        temp = []
        for each in MFCC_feature:
             temp.append(normalization(each, -1, 1))
        MFCC_feature = pandas.DataFrame(temp)   
        MFCC_feature = sta_feature(MFCC_feature)
        #   加上标签
        tag = wav_info['机组状态'] 
        MFCC_feature.append(tag)
        feature_list.append(MFCC_feature)
    pass
    # 将结果转换成易于处理的DataFrame
    data_source = pandas.DataFrame(np.array(feature_list),columns = column_list)
    # 将结果保存成excel格式，下次直接读入excel
    wbw = pandas.ExcelWriter('data_source.xlsx')
    data_source.to_excel(wbw,'data_source')
    wbw.save()
    print('程序总耗时：%d 秒'%(time() - t_start))