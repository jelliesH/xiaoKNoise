# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:40:19 2016
定义一个类，这个类的各个方法就是获取各个feature属性
@author: 390672
"""
import numpy as np
import pandas as pd
import warnings
from scipy import signal
import matplotlib.pyplot as pl

def feature_fft(wave_data, framerate):
    ##############做fft##################
    # 获取抽样的时间步长
    timestep = 1/framerate
    # 做fft
    wave_sp = np.fft.fft(wave_data)
    wave_freq = np.fft.fftfreq(wave_sp.size,d = timestep)
     
    # 只取正半轴的数据
    N = int(np.floor(wave_sp.size/2))
    # 由于量级太大，需要再缩小
    #sp = wave_sp/wave_sp.size
    sp = np.abs(wave_sp[:N])
    freq = wave_freq[:N] 
    return sp, freq
    
"""
对象 - 特征集 1 & 2
～～～～～～～～～～～～～～～～～～～～～～～～～～～
从采集到的时域信号和频谱中提取11个时域特征参数和13个
频域特征参数
"""        
class Feature(object):
    def __init__(self, wave, framerate):
        self.wave = wave
        self.framerate = framerate
        self.sp, self.freq = feature_fft(self.wave, self.framerate)
        self.freq = self.freq     
        
    def feature_p1(self):
        wave = self.wave
        p1 = wave.sum()/wave.size
        return p1

    def feature_p2(self):
        wave = self.wave
        p1 = self.feature_p1()
        var1 = np.square(wave - p1)
        
        #print('p2中的根号内值%f'%(var1.sum()/(wave.size - 1)))
        p2 = np.sqrt(var1.sum()/(wave.size - 1))
        return p2

    def feature_p3(self):
        wave = self.wave
        var1 = np.abs(wave)
        
        #print('p3中的根号内值:\n');print (var1)
        var2 = np.sqrt(var1)
        p3 = np.square(var2.sum()/wave.size)
        return p3

    def feature_p4(self ):
        wave = self.wave
        var1 = np.square(wave)
        
        #print('p4中的根号内值%f'%(var1.sum()/wave.size))
        p4 = np.sqrt(var1.sum()/wave.size)
        return p4

    def feature_p5(self ):
        wave = self.wave
        p5 = np.max(np.abs(wave))
        return p5

    def feature_p6(self ):
        wave = self.wave
        p1 = self.feature_p1()
        p2 = self.feature_p2()
        var1 = (wave-p1)**3
        p6 = var1.sum()/((wave.size - 1) * (p2**3))
        return p6

    def feature_p7(self):
        wave = self.wave
        p1 = self.feature_p1()
        p2 = self.feature_p2()       
        
        var1 = (wave - p1)**4
        p7 = var1.sum()/((wave.size - 1)*(p2**4))
        return p7

    def feature_p8(self):
        p5 = self.feature_p5()
        p4 = self.feature_p4()
        
        p8 = p5/p4
        return p8

    def feature_p9(self):
        p5 =self.feature_p5()
        p3 = self.feature_p3()
        
        p9 = p5/p3
        return p9

    def feature_p10(self):
        wave = self.wave
        p4 = self.feature_p4()
        
        var1 = np.abs(wave)
        p10 = p4/(var1.sum()/wave.size)
        return p10

    def feature_p11(self):
        wave = self.wave
        p5 = self.feature_p5()        
        
        var1 = np.abs(wave)
        p11 = p5/(var1.sum()/wave.size)
        return p11

    def feature_p12(self):
        sp = self.sp
        
        p12 = sp.sum()/sp.size
        return p12

    def feature_p13(self):
        sp = self.sp
        p12 = self.feature_p12()
        
        var1 = np.square(sp - p12)
        p13 = var1.sum()/(var1.size - 1)
        return p13

    def feature_p14(self):
        sp = self.sp
        p12 = self.feature_p12()
        p13 = self.feature_p13()
        
        var1 = (sp - p12)**3
        
        #print('p14中的根号内值%f'%(p13))
        var2 = np.sqrt(p13)
        p14 = var1.sum()/(sp.size*(var2**3))
        return p14

    def feature_p15(self):
        sp = self.sp
        p12 = self.feature_p12()
        p13 = self.feature_p13()
        
        var1 = (sp - p12)**4
        p15 = var1.sum()/(sp.size*(p13**2))
        return p15

    def feature_p16(self):
        sp = self.sp
        freq = self.freq
 
        var1 = freq*sp
        
        p16 = var1.sum()/sp.sum()
        return p16

    def feature_p17(self):
        sp = self.sp
        freq = self.freq
        p16 = self.feature_p16()       
        
        var1 = np.square(freq-p16)*sp
        
        #print('p17中的根号内值%f'%(var1.sum()/sp.size))
        p17 = np.sqrt(var1.sum()/sp.size)
        return p17

    def feature_p18(self):
        sp = self.sp
        freq = self.freq
        
        var1 = (freq**2)*sp
        
        #print('p18中的根号内值%d'%(var1.sum()/sp.sum()))
        p18 = np.sqrt(var1.sum()/sp.sum())
        return p18

    def feature_p19(self):
        sp = self.sp
        freq = self.freq       
        
        var1 = (freq**4)*sp
        var2 = (freq**2)*sp
        
        #print('p19中的根号内值%f'%(var1.sum()/var2.sum()))
        p19 = np.sqrt(var1.sum()/var2.sum())
        return p19

    def feature_p20(self):
        sp = self.sp
        freq = self.freq
        
        var1 = (freq**2)*sp
        var2 = (freq**4)*sp
        var3 = var2.sum()*sp.sum()
        #print('p20中的根号内值%f'%var3.sum())
        p20 = var1.sum()/np.sqrt(var3)
        return p20

    def feature_p21(self):
        p17 = self.feature_p17()
        p16 = self.feature_p16()
        
        p21 = p17/p16
        return p21

    def feature_p22(self):
        sp = self.sp
        freq = self.freq
        p16 = self.feature_p16()
        p17 = self.feature_p17()
        
        var1 = ((freq - p16)**3)*sp
        p22 = var1.sum()/(sp.size * (p17**3))
        return p22

    def feature_p23(self):
        sp = self.sp
        freq = self.freq
        p16 = self.feature_p16()
        p17 = self.feature_p17()
        
        var1 = ((freq - p16)**4)*sp
        p23 = var1.sum()/(sp.size * (p17**4))
        return p23

    def feature_p24(self):
        warnings.simplefilter(action = "ignore", category = RuntimeWarning)
        sp = self.sp
        freq = self.freq
        p16 = self.feature_p16()
        p17 = self.feature_p17()
        
        var1 = ((freq - p16)**0.5)*sp
        # 因为开方导致了一些nan值存在（因为根号内是负数）
        
        var1 = pd.Series(var1)
        var1 = var1.dropna()
        var1 = var1.values
        
        p24 = var1.sum()/(var1.size * (p17**0.5))
        return p24
    
    def get_time_feature(self,cluster_name = 'cl1'):
        # 获取所有时域的特征        
        feature = {
        cluster_name + '_p1':self.feature_p1(),
        cluster_name + '_p2':self.feature_p2(),
        cluster_name + '_p3':self.feature_p3(),
        cluster_name + '_p4':self.feature_p4(),
        cluster_name + '_p5':self.feature_p5(),
        cluster_name + '_p6':self.feature_p6(),
        cluster_name + '_p7':self.feature_p7(),
        cluster_name + '_p8':self.feature_p8(),
        cluster_name + '_p9':self.feature_p9(),
        cluster_name + '_p10':self.feature_p10(),
        cluster_name + '_p11':self.feature_p11()
        }        
        return feature

    def get_freq_feature(self,cluster_name = 'cl1'):
        # 获取所有频域的特征        
        feature = {
        cluster_name + '_p12':self.feature_p12(),
        cluster_name + '_p13':self.feature_p13(),
        cluster_name + '_p14':self.feature_p14(),
        cluster_name + '_p15':self.feature_p15(),
        cluster_name + '_p16':self.feature_p16(),
        cluster_name + '_p17':self.feature_p17(),
        cluster_name + '_p18':self.feature_p18(),
        cluster_name + '_p19':self.feature_p19(),
        cluster_name + '_p20':self.feature_p20(),
        cluster_name + '_p21':self.feature_p21(),
        cluster_name + '_p22':self.feature_p22(),
        cluster_name + '_p23':self.feature_p23(),
        cluster_name + '_p24':self.feature_p24(),
        }        
        return feature    
        
"""
对象 - 特征集 3
～～～～～～～～～～～～～～～～～～～～～～～～～～
使用 F个滤波器（小波分解），选择包含故障主要特征信息
的滤波频带，对信号滤波处理，提取每一个频带的信号的11
个时域特征参数。
"""
class Feature_Cl3(object):
    def __init__(self,wave,framerate,sub_name,\
                fs = [0.4],fp = [0.3],gp = 1,gs = 60):
        # sub_name 格式 cl3_f1
        self.wave = wave
        self.framerate = framerate
        
        ws = 2*np.array(fs)/framerate
        self.ws = ws.tolist() #滤波器的带通频率
        wp = 2*np.array(fp)/framerate
        self.wp = wp.tolist() #滤波器的带阻频率
        
        self.gp = gp #滤波器的带通衰减
        self.gs = gs #滤波器的带阻衰减
        self.filter = {'a':[],'b':[]} #滤波器的ab值
        self.filtered_wave = wave # 经过滤波后的信号
        self.sub_name = sub_name # 特征集3有F个滤波器滤出的信号组成，因此用sub_name来表示cl3的第n个特征集
        
    def filter_ellip(self,mode = 'low',analog = 'False'):
        # 用ellip滤波器        
        # 使用的时候根据 mode 来确定是低通还是高通还是带通滤波器{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        # analog 为True 是模拟滤波器，为false是数字滤波器
        N,Wn = signal.ellipord(self.wp,
                               self.ws,
                               self.gp,
                               self.gs,
                               analog)
        b,a = signal.ellip(N,
                           self.gp,
                           self.gs,
                           Wn,
                           mode,
                           analog)
        self.filter['a'] = a ; self.filter['b'] = b
        result = signal.lfilter(b,a,self.wave)
        self.filtered_wave = result
        return result
    
    def filter_butter(self,mode = 'low',analog = 'False'):
        # 用butter滤波器        
        # 使用的时候根据 mode 来确定是低通还是高通还是带通滤波器{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        # analog 为True 是模拟滤波器，为false是数字滤波器
        N,Wn = signal.buttord(self.wp,
                               self.ws,
                               self.gp,
                               self.gs,
                               analog)
        b,a = signal.butter(N,
                           Wn,
                           mode,
                           analog)
        self.filter['a'] = a ; self.filter['b'] = b
        result = signal.lfilter(b,a,self.wave)
        self.filtered_wave = result        
        return result       
    
    def filter_cheby1(self,mode = 'low',analog = 'False'):
        # 用cheby1滤波器        
        # 使用的时候根据 mode 来确定是低通还是高通还是带通滤波器{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        # analog 为True 是模拟滤波器，为false是数字滤波器
        N,Wn = signal.cheb1ord(self.wp,
                               self.ws,
                               self.gp,
                               self.gs,
                               analog)
        b,a = signal.cheby1(N,
                            self.gp,
                            Wn,
                            mode,
                            analog)
        self.filter['a'] = a ; self.filter['b'] = b
        result = signal.lfilter(b,a,self.wave)
        self.filtered_wave = result        
        return result       
    
    def filter_cheby2(self,mode = 'low',analog = 'False'):
        # 用cheby2滤波器        
        # 使用的时候根据 mode 来确定是低通还是高通还是带通滤波器{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        # analog 为True 是模拟滤波器，为false是数字滤波器
        N,Wn = signal.cheb2ord(self.wp,
                               self.ws,
                               self.gp,
                               self.gs,
                               analog)
        b,a = signal.cheby2(N,
                            self.gp,
                            Wn,
                            mode,
                            analog)
        self.filter['a'] = a ; self.filter['b'] = b
        result = signal.lfilter(b,a,self.wave)
        self.filtered_wave = result        
        return result       
        
    def filter_plot(self):
        # 画出滤波器的形状
        w,h = signal.freqz(self.filter['b'],self.filter['a'])        
        pl.plot( w*self.framerate*0.5/np.pi,
                abs(h))
                
    def get_cl3_feature(self):
        cl3_feature = Feature(self.filtered_wave,self.framerate)
        cl3_time_feature = cl3_feature.get_time_feature(cluster_name=self.sub_name)
        return cl3_time_feature
                
"""
对象 - 特征集 4
～～～～～～～～～～～～～～～～～～～～～～～～～～～
利用 Hilbert 解调算法对滤波后的信号进行解调处理，并
计算包络信号的频谱，再提取频谱的13个特征参数。
"""
class Feature_Cl4(object):
    def __init__(self,wave,framerate,subname):
        self.wave = wave  
        self.sp = []
        self.framerate= framerate
        self.subname = subname # 特征集4有F个滤波器滤出的信号组成，因此用sub_name来表示cl4的第n个特征集
        self.amplitude_envelope = wave # 利用hilbert获取的包络信号
        
    def hilbert_envelop(self):
        # 利用 Hilbert 解调算法对滤波后的信号进行解调处理，并
        # 计算包络信号
        wave = self.wave
        analytic_signal = signal.hilbert(wave)
        amplitude_envelope = np.abs(analytic_signal)
        self.amplitude_envelope = amplitude_envelope
        return amplitude_envelope
        
    def get_cl4_feature(self):
        # 函数说明：
        #--------------------------------------        
        # 计算包络谱的频谱特征
        #   amplitude_envelope - 包络谱
        #   cl4_freq_feature - dict结构，每个元素是一个 包络 对应的13个频域特征
        #--------------------------------------    
        cl4_feature = Feature(self.amplitude_envelope,self.framerate)
        cl4_freq_feature = cl4_feature.get_freq_feature(cluster_name=self.subname)
        return cl4_freq_feature
        
"""
对象 - 特征集 5
～～～～～～～～～～～～～～～～～～～～～～～～～～～
使用 EMD 方法对原始信号进行分解，选取包含有用信息的前
8个本征模式。从8个本正模式分量中分别提取11个时域特征。
除了传入的 object 之外，engine是matlab对应的API
"""
class Feature_Cl5(object):
    def __init__(self, wave,framerate,subname):
        self.wave = wave
        self.framerate = framerate
        self.subname = subname # 特征集5有8个本征模式分量组成，因此用cl5_imfn来表示cl5的第n个特征集
        self.amplitude_envelope = wave # 利用hilbert获取的包络信号
        self.imfs = []
        
    def emd_analysis(self,eng,ml):
        # 函数说明：
        #--------------------------------------        
        # 进行EMD分解，eng是matlab 引擎
        #--------------------------------------       
        wave = self.wave
        # 将数据转换matlab的数组结构
        
        wave2 = ml.double(wave.tolist())
        print('进入EMD分解……')
        imfs = eng.emd(wave2)
        imfs_p = []
        # 将结果转换成python的数据类型
        for each in imfs:          
            imfs_p.append(eng.num2cell(each))
        imfs = np.array(imfs_p)
        self.imfs = imfs
    
    def get_cl5_feature(self):
        # 函数说明：
        #--------------------------------------        
        # 获取EMD分解后各个时域特征
        #   imfs - list结构，每个元素是一个imf 是 np.array格式
        #   result - list结构，每个元素是一个 imf 对应的11个时域特征
        #--------------------------------------    
        result = dict()
        imfs = self.imfs
        imfs_num = np.arange(len(imfs))
        # 取前8个作为特征集
        for ii in imfs_num[:8]:
            subname = self.subname + 'imf%d' % ii
            cl5_feature = Feature(imfs[ii],self.framerate)
            result[subname] = cl5_feature.get_time_feature(cluster_name = subname)
        return result, imfs[:8]
"""
对象 - 特征集 6
～～～～～～～～～～～～～～～～～～～～～～～～～～～
对 8 个本征模式分量分别进行 Hilbert 变换，并提取其包络谱
的13个频域特征。
"""
class Feature_Cl6(object):
    def __init__(self,imfs,framerate,subname):
        self.imfs = imfs
        self.subname = subname
        self.framerate = framerate
        
    def get_cl6_feature(self):
        result = dict()
        for ii in range(len(self.imfs)):
            subname = self.subname + str(ii)
            # 利用 特征4 建立 希尔波特变换的类
            f_cl6 = Feature_Cl4(self.imfs[ii],self.framerate,subname)
            # 做希尔波特变换
            f_cl6.hilbert_envelop()
            result[subname] = f_cl6.get_cl4_feature()
        # 返回结果
        return result
        