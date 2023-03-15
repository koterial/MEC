import math


# 距离转换成功率
# 输入: 功率(dbm), 距离(m)
# 输出: 功率(mw)
def get_power(power, distance):
    log_p = (power - 128.1 - 37.6 * math.log10(distance / 1000)) / 10
    power = math.pow(10, log_p)
    return power


# 功率mw转换成单位dbm
# 输入: 功率(mw)
# 输出: 功率(dbm)
def mw_to_dbm(power):
    return math.log10(power) * 10


# 功率dbm转换成单位mw
# 输入: 功率(dbm)
# 输出: 功率(mw)
def dbm_to_mw(power):
    return math.pow(10, power / 10)


# 功率转换成速率
# 输入: 功率(mw), 带宽(Hz), 噪声功率谱密度(mw/Hz)
# 输出: 速率(bit/s)
def get_speed(power, bandwidth, noise_power=math.pow(10, -17.4)):
    noise_totle_power = max(bandwidth * noise_power, 1e-18)
    SNIR = power / noise_totle_power
    speed = bandwidth * math.log2(1 + SNIR)
    return speed


# 计算两点之间的距离
# 输入: 坐标1(m), 坐标2(m)
# 输出: 距离(m)
def get_distance(xpos1, ypos1, zpos1, xpos2, ypos2, zpos2):
    xpos = xpos2 - xpos1
    ypos = ypos2 - ypos1
    zpos = zpos2 - zpos1
    return max(math.sqrt(xpos ** 2 + ypos ** 2 + zpos ** 2), 0.5)