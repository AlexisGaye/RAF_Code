import torch
import pickle
import numpy as np
import skfuzzy as fuzz
from torch import nn
import fcm_fnn
from torch.utils.data import DataLoader
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

# set up
def loadDataSet(filepath):
    with open(filepath) as f:
        rawList = map(lambda line: float(line.strip()), f.readlines())
        labelSet = []
        testSet = []
        for i in range(8, 8 + 500):
            labelSet.append((rawList[i:i+4], rawList[i+1:i+5]))
        for i in range(8 + 500, 8 + 1000):
            testSet.append((rawList[i:i+4], rawList[i+1:i+5]))
        return labelSet, testSet

# 创建数据集实例
class GreenhouseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

# 读取温湿度数据集和标签
data = loadDataSet("./data/datas2024_3_26.dat")
targets = loadDataSet('temperature','soil_humidity','air_humidity','LightIntensity','Ph','CarbonConcen')

# 划分训练集和测试集
train_data = data[:1800]
train_targets = targets[:1800]
test_data = data[200:]
test_targets = targets[200:]

# 数据增强和预处理
transform = transforms.Compose([
    transforms.ToTensor()  # 将数据转为Tensor
])

# 创建输入变量
temperature = ctrl.Antecedent(train_data.temperature, 'temperature')
soil_humidity = ctrl.Antecedent(train_data.soil_humidity, 'soil_humidity')
air_humidity = ctrl.Antecedent(train_data.air_humidity, 'air_humidity')
LightIntensity = ctrl.Antecedent(train_data.LightIntensity, 'LightIntensity')
Ph = ctrl.Antecedent(train_data.Ph, 'Ph')
CarbonConcen = ctrl.Antecedent(train_data.CarbonConcen, 'CarbonConcen')

# 定义温度隶属函数
# NB,NS,ZE,PS,PB
temperature['NB'] = fuzz.trimf(temperature.universe, [0, 0, 0, 0, 4, 11])
temperature['NS'] = fuzz.trimf(temperature.universe, [0, 0, 0, 4, 11, 15])
temperature['ZE'] = fuzz.trimf(temperature.universe, [0, 0, 4, 11, 15, 21])
temperature['PS'] = fuzz.trimf(temperature.universe, [0, 4, 11, 15, 21, 28])
temperature['PB'] = fuzz.trimf(temperature.universe, [4, 11, 15, 21, 28, 35])

# 定义soil_humidity隶属函数
# NB,NS,ZE,PS,PB
soil_humidity['NB'] = fuzz.trimf(soil_humidity.universe, [0, 0, 0, 0, 0.40, 0.45])
soil_humidity['NS'] = fuzz.trimf(soil_humidity.universe, [0, 0, 0,0.40, 0.45, 0.55])
soil_humidity['ZE'] = fuzz.trimf(soil_humidity.universe, [0, 0, 0.40, 0.45, 0.55, 0.65])
soil_humidity['PS'] = fuzz.trimf(soil_humidity.universe, [0, 0.40, 0.45, 0.55, 0.65, 0.7])
soil_humidity['PB'] = fuzz.trimf(soil_humidity.universe, [0.40, 0.45, 0.55, 0.65, 0.7, 0.8])

# 定义air_humidity隶属函数
# NB,NS,ZE,PS,PB
air_humidity['NB'] = fuzz.trimf(air_humidity.universe, [0, 0, 0, 0, 0.50, 0.55])
air_humidity['NS'] = fuzz.trimf(air_humidity.universe, [0, 0, 0, 0.50, 0.55, 0.60])
air_humidity['ZE'] = fuzz.trimf(air_humidity.universe, [0, 0, 0.50, 0.55, 0.60, 0.65])
air_humidity['PS'] = fuzz.trimf(air_humidity.universe, [0, 0.50, 0.55, 0.60, 0.65, 0.7])
air_humidity['PB'] = fuzz.trimf(air_humidity.universe, [0.50, 0.55, 0.60, 0.65, 0.7, 0.8])

# 定义LightIntensity隶属函数
# N,Z,P
LightIntensity['N'] = fuzz.trimf(LightIntensity.universe, [0, 0, 0])
LightIntensity['Z'] = fuzz.trimf(LightIntensity.universe, [0, 60000 ,0])
LightIntensity['P'] = fuzz.trimf(LightIntensity.universe, [0, 60000, 100000])

# 定义Ph隶属函数
# NB,NM,NS,ZE,PS,PM,PB
Ph['NB'] = fuzz.trimf(Ph.universe, [0, 0, 0, 0, 4.5, 5.5])
Ph['NS'] = fuzz.trimf(Ph.universe, [0, 0, 0, 4.5, 5.5, 6.5])
Ph['ZE'] = fuzz.trimf(Ph.universe, [0, 0, 4.5, 5.5, 6.5, 7.5])
Ph['PS'] = fuzz.trimf(Ph.universe, [0, 4.5, 5.5, 6.5, 7.5, 8.5])
Ph['PB'] = fuzz.trimf(Ph.universe, [4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

# 定义CarbonConcen隶属函数
# NB,NS,ZE,PS,PB
CarbonConcen['NB'] = fuzz.trimf(CarbonConcen.universe, [0, 0, 0, 0, 0, 400])
CarbonConcen['NS'] = fuzz.trimf(CarbonConcen.universe, [0, 0, 0, 0, 400, 800])
CarbonConcen['ZE'] = fuzz.trimf(CarbonConcen.universe, [0, 0, 0, 400, 800, 1200])
CarbonConcen['PS'] = fuzz.trimf(CarbonConcen.universe, [0, 0, 400, 800, 1200, 1600,])
CarbonConcen['PB'] = fuzz.trimf(CarbonConcen.universe, [0, 400, 800, 1200, 1600, 2000])

Water =[0,1,2,3,4]
change_carbonConcen =[0,1,2,3,4]
change_temperature =[0,1,2,3,4]
change_humidity =[0,1,2,3,4]
change_lightIntensity =[0,1,2,3,4]


rule1 = ctrl.Rule((temperature['NB'] or temperature['NS']) & soil_humidity['NB'] & (air_humidity['NB'] or air_humidity['NS']) & LightIntensity['N'] & Ph['NB'] & CarbonConcen['NB'], water[2] & change_carbonConcen[3] & change_temperature[4] & change_humidity[2] & change_lightIntensity[4])
rule2 = ctrl.Rule((temperature['NS'] or temperature['ZE']) & soil_humidity['NB'] & air_humidity['PB'] & LightIntensity['N'] & Ph['NB'] & CarbonConcen['NS'], water[4] & change_carbonConcen[2] & change_temperature[2] & change_humidity[2] & change_lightIntensity[1])
rule3 = ctrl.Rule((temperature['NS'] or temperature['ZE']) & soil_humidity['NS'] & air_humidity['PS'] & LightIntensity['Z'] & Ph['NB'] & CarbonConcen['NS'], water[3] & change_carbonConcen[3] & change_temperature[3] & change_humidity[2] & change_lightIntensity[2])
rule4 = ctrl.Rule(temperature['ZE'] & soil_humidity['NB'] & air_humidity['PB'] & LightIntensity['P'] & Ph['NB'] & (CarbonConcen['NB'] or CarbonConcen['NS']), water[1] & change_carbonConcen[3] & change_temperature[1] & change_humidity[2] & change_lightIntensity[2])
rule5 = ctrl.Rule(temperature['PS'] & soil_humidity['NB'] & air_humidity['PS'] & LightIntensity['Z'] & Ph['NB'] & CarbonConcen['PS'], water[2] & change_carbonConcen[3] & change_temperature[1] & change_humidity[2] & change_lightIntensity[1])
rule6 = ctrl.Rule(temperature['PB'] & soil_humidity['NB'] & air_humidity['ZE'] & LightIntensity['P'] & Ph['NB'] & CarbonConcen['PB'], water[2] & change_carbonConcen[3] & change_temperature[2] & change_humidity[2] & change_lightIntensity[3])


# 创建控制器
water_ctrl = ctrl.ControlSystem(Max([rule1[0], rule2[0], rule3[0], rule4[0] , rule5[0] , rule6[0]])
co2_ctrl = ctrl.ControlSystem(Max([rule1[1], rule2[1], rule3[1], rule4[1] , rule5[1] , rule6[1]]))
tem_ctrl = ctrl.ControlSystem(Max([rule1[2], rule2[2], rule3[2], rule4[2] , rule5[2] , rule6[2]]))
hum_ctrl = ctrl.ControlSystem(Max[rule1[3], rule2[3], rule3[3], rule4[3] , rule5[3] , rule6[3]])
light_ctrl = ctrl.ControlSystem(Max[rule1[4], rule2[4], rule3[4], rule4[4] , rule5[4] , rule6[4]])

# 创建控制器模拟器
water_sim = ctrl.ControlSystemSimulation(water_ctrl)
co2_sim = ctrl.ControlSystemSimulation(co2_ctrl)
tem_sim = ctrl.ControlSystemSimulation(tem_ctrl)
hum_sim = ctrl.ControlSystemSimulation(hum_ctrl)                           
light_sim = ctrl.ControlSystemSimulation(light_ctrl)
                                
# 输入模糊推理系统
water_sim.input['temperature'] = train_data.temperature
water_sim.input['soil_humidity'] = train_data.soil_humidity
water_sim.input['air_humidity'] = train_data.air_humidity
water_sim.input['LightIntensity'] = train_data.LightIntensity
water_sim.input['Ph'] = train_data.Ph
water_sim.input['CarbonConcen'] = train_data.CarbonConcen
                                
co2_sim.input['temperature'] = train_data.temperature
co2_sim.input['soil_humidity'] = train_data.soil_humidity
co2_sim.input['air_humidity'] = train_data.air_humidity
co2_sim.input['LightIntensity'] = train_data.LightIntensity
co2_sim.input['Ph'] = train_data.Ph
co2_sim.input['CarbonConcen'] = train_data.CarbonConcen
                                
tem_sim.input['temperature'] = train_data.temperature
tem_sim.input['soil_humidity'] = train_data.soil_humidity
tem_sim.input['air_humidity'] = train_data.air_humidity
tem_sim.input['LightIntensity'] = train_data.LightIntensity
tem_sim.input['Ph'] = train_data.Ph
tem_sim.input['CarbonConcen'] = train_data.CarbonConcen
                                
hum_sim.input['temperature'] = train_data.temperature
hum_sim.input['soil_humidity'] = train_data.soil_humidity
hum_sim.input['air_humidity'] = train_data.air_humidity
hum_sim.input['LightIntensity'] = train_data.LightIntensity
hum_sim.input['Ph'] = train_data.Ph
hum_sim.input['CarbonConcen'] = train_data.CarbonConcen
                                
light_sim.input['temperature'] = train_data.temperature
light_sim.input['soil_humidity'] = train_data.soil_humidity
light_sim.input['air_humidity'] = train_data.air_humidity
light_sim.input['LightIntensity'] = train_data.LightIntensity
light_sim.input['Ph'] = train_data.Ph
light_sim.input['CarbonConcen'] = train_data.CarbonConcen
                                
# 运行模糊推理
water_sim.compute()
co2_sim.compute()
tem_sim.compute()
hum_sim.compute()
light_sim.compute()

# 获取训练集
output_f=[water_sim.output['water'],co2_sim.output['co2'],tem_sim.output['tem'],hum_sim.output['hum'],light_sim.output['light']]
train_data =output_f

# 重置输出测试集
# 创建控制器模拟器
water_sim = ctrl.ControlSystemSimulation(water_ctrl)
co2_sim = ctrl.ControlSystemSimulation(co2_ctrl)
tem_sim = ctrl.ControlSystemSimulation(tem_ctrl)
hum_sim = ctrl.ControlSystemSimulation(hum_ctrl)                           
light_sim = ctrl.ControlSystemSimulation(light_ctrl)
                                
# 设置输入值
water_sim.input['temperature'] = test_data.temperature
water_sim.input['soil_humidity'] = test_data.soil_humidity
water_sim.input['air_humidity'] = test_data.air_humidity
water_sim.input['LightIntensity'] = test_data.LightIntensity
water_sim.input['Ph'] = test_data.Ph
water_sim.input['CarbonConcen'] = test_data.CarbonConcen
                                
co2_sim.input['temperature'] = test_data.temperature
co2_sim.input['soil_humidity'] = test_data.soil_humidity
co2_sim.input['air_humidity'] = test_data.air_humidity
co2_sim.input['LightIntensity'] = test_data.LightIntensity
co2_sim.input['Ph'] = test_data.Ph
co2_sim.input['CarbonConcen'] = test_data.CarbonConcen
                                
tem_sim.input['temperature'] = test_data.temperature
tem_sim.input['soil_humidity'] = test_data.soil_humidity
tem_sim.input['air_humidity'] = test_data.air_humidity
tem_sim.input['LightIntensity'] = test_data.LightIntensity
tem_sim.input['Ph'] = test_data.Ph
tem_sim.input['CarbonConcen'] = test_data.CarbonConcen
                                
hum_sim.input['temperature'] = test_data.temperature
hum_sim.input['soil_humidity'] = test_data.soil_humidity
hum_sim.input['air_humidity'] = test_data.air_humidity
hum_sim.input['LightIntensity'] = test_data.LightIntensity
hum_sim.input['Ph'] = test_data.Ph
hum_sim.input['CarbonConcen'] = test_data.CarbonConcen
                                
light_sim.input['temperature'] = test_data.temperature
light_sim.input['soil_humidity'] = test_data.soil_humidity
light_sim.input['air_humidity'] = test_data.air_humidity
light_sim.input['LightIntensity'] = test_data.LightIntensity
light_sim.input['Ph'] = test_data.Ph
light_sim.input['CarbonConcen'] = test_data.CarbonConcen
                                
# 运行模糊推理
water_sim.compute()
co2_sim.compute()
tem_sim.compute()
hum_sim.compute()
light_sim.compute()

# 获取输出值
output_f=[water_sim.output['water'],co2_sim.output['co2'],tem_sim.output['tem'],hum_sim.output['hum'],light_sim.output['light']]
test_data =output_f
                                
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
                                

# 创建网络模型
class Raf(nn.Module):
    def __init__(self):
        super(Raf, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 12),  
            nn.ReLU(),
            nn.Linear(12, 5)  
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
                                
raf = Raf()
raf = raf.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
"""
learning_rate = 1e-2
optimizer = torch.optim.SGD(raf.parameters(), lr=learning_rate)
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    raf.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = raf(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    raf.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = raf(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(raf, "raf_{}.pth".format(i))
    print("模型已保存")

writer.close()
