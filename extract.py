import os
import xarray as xr
import numpy as np
import pandas as pd


def calculate_wind_components(wrf_file, output_file):
    """
    将WRF输出的U10/V10合成风向风速
    Args:
        wrf_file: 输入的wrfout文件路径
        output_file: 输出的新文件路径
    """
    try:
        # 打开WRF文件（自动解析维度）
        ds = xr.open_dataset(wrf_file, engine= 'netcdf4')

        # 检查变量是否存在
        required_vars = ['U10', 'V10']
        for var in required_vars:
            if var not in ds:
                raise ValueError(f"文件缺少必要变量: {var}")

        # 计算风速（m/s）
        with xr.set_options(keep_attrs=True):  # 保留原始属性
            u = ds['U10']
            v = ds['V10']

            # 计算风速（避免除零错误）
            wind_speed = np.sqrt(u**2 + v**2)
            wind_speed.attrs.update({
                'long_name': 'Wind Speed at 10m',
                'units': 'm s⁻¹',
                'standard_name': 'wind_speed'
            })

            # 计算风向（度数，0-360）
            # 使用arctan2计算角度（弧度转度数）
            angle_rad = np.arctan2(-u, -v)  # WRF坐标系调整（东→北转为标准数学坐标）
            wind_dir = np.degrees(angle_rad) % 360  # 转换为0-360度

            wind_dir.attrs.update({
                'long_name': 'Wind Direction at 10m',
                'units': 'degrees',
                'standard_name': 'wind_from_direction',
                'comment': 'Direction from which wind is blowing'
            })

        # 创建新数据集
        new_ds = xr.Dataset({
            'wind_speed': wind_speed,
            'wind_dir': wind_dir
        })

        # 复制原始文件的全局属性
        new_ds.attrs.update(ds.attrs)

        # 保存文件（启用压缩）
        encoding = {
            'wind_speed': {'zlib': True, 'complevel': 5},
            'wind_dir': {'zlib': True, 'complevel': 5}
        }
        new_ds.to_netcdf(output_file, engine='netcdf4', encoding=encoding)

        print(f"成功生成文件：{output_file}")
        print("新变量属性示例：")
        print(new_ds['wind_speed'].attrs)
        print(new_ds['wind_dir'].attrs)

    except Exception as e:
        print(f"处理失败：{str(e)}")
    finally:
        ds.close()

def calculate_mean(file_path):
    ds = xr.open_dataset(file_path, engine= 'netcdf4')

    with xr.set_options(keep_attrs=True):
        wind_dir = ds['wind_dir']
        wind_speed = ds['wind_speed']

        dir_mean = np.nanmean(wind_dir)
        speed_mean = np.nanmean(wind_speed)
    return dir_mean, speed_mean

if "__main__" == __name__:
    path = "D:/WRFOUT_dataprocess/dataset/EXTRACT_d2"
    files = os.listdir(f"{path}/")
    # for file in files:
    #     calculate_wind_components(
    #         wrf_file= path+"/"+file,
    #         output_file=f'EXTRACT/EXTRACT_{file}.nc',
    #     )
    df = pd.DataFrame()
    for file in files:
        dir_mean, speed_mean = calculate_mean(f"{path}/{file}")
        new_row = pd.DataFrame([[dir_mean, speed_mean]], columns=['dir_mean', 'speed_mean'])
        df = pd.concat([df, new_row])

    df.to_excel("ouput2.xlsx", index=False)