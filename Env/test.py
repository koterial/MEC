import json
import numpy as np

if __name__ == "__main__":
    system_info = {
        "mec_system": {
            # 世界坐标系(m)
            "map_width_size": 1000,
            "map_height_size": 1000,
            # 像素坐标系(pixel)
            "screen_width_size": 1000,
            "screen_height_size": 1000,
            # 时隙长度(s)
            "tau": 1 * 1e-2,
            # 最大运行时间(时隙数量)
            "max_run_time": 100,
        },
        "device_info_list": [{
            # 设备类别
            "type": 0,
            # 设备名称
            "name": "edge_server",
            # 最大任务生成数量
            "max_task_generation_num": 0,
            # 最大任务承载数量
            "max_task_load_num": 5,
            # 单个任务最大运行时间(时隙数量)
            "max_task_run_time": 50,
            # 单个任务最大上传数据量(bit)
            "max_task_upload_data_size": 0,
            # 单个任务最大存储数据量(bit)
            "max_task_store_data_size": 0,
            # 单个任务最大计算数据量(cycle)
            "max_task_compute_data_size": 0,
            # 单个任务最大下载数据量(bit)
            "max_task_download_data_size": 0,
            # 设备单路信号最大发射功率(dbm)
            "max_transmission_power": 30,
            # 设备最大计算功率(dbm)
            "max_compute_power": 1,
            # 设备最大待机功率(dbm)
            "max_standby_power": 1,
            # 设备最大上行链路带宽(Hz)
            "max_upload_resources": 1 * 1e7,
            # 设备最大缓存能力(bit)
            "max_store_resources": 0,
            # 设备最大计算能力(Hz)
            "max_compute_resources": 1 * 1e10,
            # 设备最大下行链路带宽(Hz)
            "max_download_resources": 5 * 1e6,
            # 设备最大速度(m/s)
            "max_velocity": 0,
            # 设备最大链接范围(m)
            "max_coverage": 1500
        },
        {
            "type": 1,
            "name": "vehicle",
            "max_task_generation_num": 1,
            "max_task_load_num": 0,
            "max_task_run_time": 50,
            "max_task_upload_data_size": 3 * 1e6,
            "max_task_store_data_size": 0,
            "max_task_compute_data_size": 2 * 1e8,
            "max_task_download_data_size": 1.5 * 1e6,
            "max_transmission_power": 27,
            "max_compute_power": 1,
            "max_standby_power": 20,
            "max_upload_resources": 0,
            "max_store_resources": 0,
            "max_compute_resources": 0,
            "max_download_resources": 0,
            "max_velocity": 0,
            "max_coverage": 1500
        }]
    }

    for device in system_info["device_info_list"]:
        device["max_connect_num_list"] = np.zeros(shape=(len(system_info["device_info_list"])), dtype=np.int8).tolist()
        device["max_received_power_list"] = np.zeros(shape=(len(system_info["device_info_list"])), dtype=np.int8).tolist()
    system_info["mec_system"]["device_num_list"] = np.zeros(shape=(len(system_info["device_info_list"])), dtype=np.int8).tolist()

    with open("system_info.json", 'w') as f:
        json.dump(system_info, f)

    # with open("system_info.json", 'r') as f:
    #     system_info = json.load(f)
    # print(system_info)
