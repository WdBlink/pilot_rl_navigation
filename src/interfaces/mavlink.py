import subprocess
import socket
import time
import os
from pymavlink import mavutil
import math
import inspect
# import keyboard
import matplotlib
matplotlib.use('Agg')  # 设置为非GUI后端，避免Qt错误
import matplotlib.pyplot as plt
import threading
from datetime import datetime
# import serial

def calculate_checksum(nmea_str):
    """计算NMEA语句的校验和"""
    checksum = 0
    for char in nmea_str[1:]:  # 跳过 $ 符号
        if char == '*':
            break
        checksum ^= ord(char)
    return format(checksum, '02X')  # 返回两位十六进制

def format_nmea_degrees(degrees):
    """将十进制经纬度转换为NMEA格式 (DDMM.MMMM)"""
    deg = int(degrees)
    min = (degrees - deg) * 60
    integer_part = int(min)
    decimal_part = min - integer_part
    min_str = f"{integer_part:02d}.{decimal_part * 100000:05.0f}"
    return f"{deg:02d}{min_str}"

def calculate_bearing(lat1, lon1, lat2, lon2):
    # 将经纬度从度数转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    initial_bearing = math.atan2(x, y)

    # 转换成度，并归一化到 0~360
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
class CustomSITL:
    def __init__(self,config,logger):
        self.mav_connection = None
        self.gps_serial=None
        self.args=config['mavlink']
        self.logger=logger

        # 模拟遥控器信号
        self.rc_values = [65535] * 18
        self.turn_rate = 20
        self.turn_angle = 60
        self.current_yaw = 90  # 存储当前yaw值


        self.last_alt = 0  # 存储上一点高度
        self.last_lat = 0  # 存储上一点纬度
        self.last_lon = 0  # 存储上一点经度```````
        self.EARTH_RADIUS = 6378137  # 地球半径(米)
        # 用last与current计算航向与速度，目前未使用

        # 存储当前高度,GPS高度,气压计高度
        self.global_position_alt = 0  
        self.press_alt=0
        
        self.real_lat = 0  # 存储simstate真实纬度
        self.real_lon = 0  # 存储simstate真实经度
        self.global_position_lat=0 #存储global_position_int中飞控计算纬度
        self.global_position_lon=0 #存储global_position_int中飞控计算经度
        self.vn, self.ve, self.vd = 0, 0, 0  # 速度信息
        self.pitch, self.roll, self.yaw = 0, 0, 0  # 姿态信息,弧度制，范围为-3.14~+3.14


        #气压计温度计以计算高程的相关变量
        self.press_origin = 0
        self.temp_origin = 0
        self.alt_origin=0

    def connect_to_sitl(self):
        """连接到 SITL"""
        try:
            self.mav_connect = self.args["mav_connect"]
            if self.mav_connect == "tcp":
                print("尝试通过 TCP 建立连接...")
                try:
                    print(f'tcp:{self.args["tcp_ip"]}:5762')
                    self.mav_connection = mavutil.mavlink_connection(f'tcp:{self.args["tcp_ip"]}:5762')
                    self.mav_connection.wait_heartbeat(timeout=5)
                    print("成功通过 TCP 连接 SITL")                    
                except Exception as e:
                    print(f"TCP 连接失败: {e}")
                    return False
            elif self.mav_connect == "uart":
                try:
                    print("尝试通过 Jetson 串口连接")
                    self.mav_connection = mavutil.mavlink_connection(self.args["mav_ser_name"], baud=115200)
                    self.mav_connection.wait_heartbeat(timeout=3)
                    print("成功通过串口连接，该串口将用于传输mavlink信息")
                except Exception as e:
                    print(f"mavlink通信串口连接失败: {str(e)}")
                    return False

            self.mav_connection.wait_heartbeat(timeout=5)
            print("成功获取心跳")
            if self.args["gps_send"] == "nmea":
                try:
                    self.gps_serial = serial.Serial(self.args["gps_ser_name"], baudrate=115200, timeout=1)
                    self.gps_serial.flushInput()  # 清空输入缓冲区
                    print("[INFO] GPS串口连接成功，该串口将用于传输NMEA数据")
                except (serial.SerialException, FileNotFoundError) as e:
                    self.gps_serial = None
                    print(f"[WARNING] 串口 {self.args['gps_ser_name']} 打开失败：{e}，跳过串口发送。")
            else:
                print("[INFO]使用mavlink发送GPS数据")
                self.gps_serial = None

            # 设置超时
            self.mav_connection.source_system = 255
            self.mav_connection.source_component = 0

            """禁用起飞前检查"""
            self.mav_connection.mav.param_set_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                b'ARMING_CHECK',  # 参数名
                0,  # 设置为0表示禁用所有检查
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            # 等待参数设置确认
            time.sleep(1)
            print("已禁用起飞前检查")
            print("成功连接到 SITL")
            
            # 确保系统ID和组件ID已正确设置
            print(f"System ID: {self.mav_connection.target_system}")
            print(f"Component ID: {self.mav_connection.target_component}")

            #顺便进行气压计和温度计的初始状态记录
            while True:
                msg=self.mav_connection.recv_msg()
                if msg is None:
                    break
            try:
                self.press_origin=self.mav_connection.messages['SCALED_PRESSURE'].press_abs
                self.temp_origin=self.mav_connection.messages['SCALED_PRESSURE'].temperature/100
                self.alt_origin=self.mav_connection.messages['GLOBAL_POSITION_INT'].alt/1000
                self.logger.log(f"气压计初始值：{self.press_origin}，温度计初始值：{self.temp_origin}，高度计初始值：{self.alt_origin}")
                return True
            except Exception as e:
                self.logger.log(f"接收mavlink消息失败: {str(e)}")
                return False

        except Exception as e:
            print(f"连接 SITL 失败: {str(e)}")
            return False

    def stop_sitl(self):
        """停止 SITL"""
        if self.sitl_process:
            self.sitl_process.terminate()
            print("SITL 已停止")
        if self.mav_connection:
            self.mav_connection.close()

    def SIM_GPS_DISABLE(self,value):
        self.mav_connection.mav.param_set_send(
            self.mav_connection.target_system,
            self.mav_connection.target_component,
            b'SIM_GPS_DISABLE',
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )

    def SIM_RC_FAIL(self,value):
        self.mav_connection.mav.param_set_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                b'SIM_RC_FAIL',
                value,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
    def wait_for_message(self, message_type, timeout, seq=None):
        """等待特定类型的消息，带序号检查"""
        # present heading: VFR_HUD {airspeed : 25.004688262939453, groundspeed : 0.0, heading : 87, throttle : 38, alt : 128.88999938964844, climb : 0.7966185212135315}
        start = time.time()
        while time.time() - start < timeout:
            msg = self.mav_connection.recv_match(type=message_type, blocking=True, timeout=1)
            if msg:
                if seq is None or (hasattr(msg, 'seq') and msg.seq == seq):
                    return msg
                    # 清理其他消息
            while self.mav_connection.recv_match(blocking=False):
                pass
        return None

    def upload_mission(self, waypoints):
        try:
            print("开始上传任务...")

            # 清除现有任务并确认
            print("清除现有任务...")
            for _ in range(3):  # 尝试3次
                self.mav_connection.waypoint_clear_all_send()
                msg = self.mav_connection.recv_match(type=['MISSION_ACK'], blocking=True, timeout=2)
                if msg:
                    print("清除成功")
                    break
            print(f"发送航点数量: {len(waypoints)}")
            self.mav_connection.mav.mission_count_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                len(waypoints)
            )
            # 上传航点
            for i in range(len(waypoints)):
                success = False
                for retry in range(3):  # 每个航点尝试3次
                    # 等待请求，过滤掉其他消息
                    start_time = time.time()
                    while time.time() - start_time < 5:  # 5秒超时
                        msg = self.wait_for_message(['MISSION_REQUEST_INT', 'MISSION_REQUEST'], 1)
                        if not msg:
                            print(f"未收到航点 {i} 的请求，重试...")
                        if msg and (
                                msg.get_type() == 'MISSION_REQUEST_INT' or msg.get_type() == 'MISSION_REQUEST') and msg.seq == i:
                            # 收到正确的请求，发送航点
                            self.mav_connection.mav.mission_item_int_send(
                                self.mav_connection.target_system,
                                self.mav_connection.target_component,
                                i,
                                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                0, 1,
                                0, 0, 0, 0,
                                int(waypoints[i]['lat'] * 1e7),  # 转换为整数坐标
                                int(waypoints[i]['lon'] * 1e7),
                                waypoints[i]['alt']
                            )
                            print(f"发送航点 {i}")
                            success = True
                            break

                    if success:
                        break
                    else:
                        print(f"重试发送航点 {i}")
                        # 重新发送航点数量
                        self.mav_connection.mav.mission_count_send(
                            self.mav_connection.target_system,
                            self.mav_connection.target_component,
                            len(waypoints)
                        )
                        time.sleep(1)

                if not success:
                    print(f"航点 {i} 上传失败")
                    return False

                time.sleep(0.1)  # 短暂延时

            # 等待最终确认
            print("等待最终确认...")
            for _ in range(3):
                msg = self.mav_connection.recv_match(type=['MISSION_ACK'], blocking=True, timeout=2)
                if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    print("任务上传完成并确认")
                    break
                time.sleep(1)

            # 验证任务
            print("开始验证任务...")
            self.mav_connection.waypoint_request_list_send()
            msg = self.mav_connection.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=2)
            if not msg or msg.count != len(waypoints):
                print(f"验证失败: 预期 {len(waypoints)} 个航点，实际 {msg.count if msg else 0} 个")
                return False

            print(f"验证成功: 共 {msg.count} 个航点")

            self.set_mode("AUTO")
            return True

        except Exception as e:
            print(f"错误: {str(e)}")
            return False

    def set_mode(self, mode, timeout=20):
        """设置飞行模式"""
        mode_map = {
            'STABILIZE': 2,
            'FBWA': 5,
            'AUTO': 10,
            'GUIDED': 15,
            'LOITER': 12,
            'RTL': 11,
            'FBWB': 6,
        }

        if mode not in mode_map:
            print(f"不支持的模式: {mode}")
            return False

        # 使用 command_long_send 来设置模式
        self.mav_connection.mav.command_long_send(
            self.mav_connection.target_system,
            self.mav_connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_map[mode],
            0, 0, 0, 0, 0
        )

        # 等待模式切换确认，添加超时机制
        start_time = time.time()
        while time.time() - start_time < timeout:
            while True:
                m=self.mav_connection.recv_msg()
                if m is None:
                    break
            try:
                msg=self.mav_connection.messages['HEARTBEAT']
                if msg:
                    current_mode = msg.custom_mode
                    print(f"当前模式: {current_mode}, 目标模式: {mode_map[mode]}")
                    if current_mode == mode_map[mode]:
                        print(f"成功切换到 {mode} 模式")
                        return True
                    else:
                        self.mav_connection.mav.command_long_send(
                            self.mav_connection.target_system,
                            self.mav_connection.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                            0,  # confirmation
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            mode_map[mode],
                            0, 0, 0, 0, 0
                        )
                        time.sleep(0.5)
                else:
                    print("未收到 HEARTBEAT 消息")
            except Exception as e:
                print(f"接收消息时出错: {str(e)}")

        print(f"切换到 {mode} 模式超时")
        return False

    def arm_vehicle(self):
        """解锁并起飞到指定高度"""
        print("开始解锁和起飞程序...")

        # 解锁
        print("发送解锁命令")
        self.mav_connection.mav.command_long_send(
            self.mav_connection.target_system,
            self.mav_connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.mav_connection.recv_match(type='COMMAND_ACK', blocking=True,timeout=1)
        if ack_msg:
            print(f"解锁命令响应: command={ack_msg.command}, result={ack_msg.result}")
        # 等待解锁确认
        start_time = time.time()
        armed = False
        while time.time() - start_time < 10:  # 10秒超时
            msg = self.mav_connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                armed = True
                print("飞机已解锁")
                break
            else:
                ack_msg = self.mav_connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
                if ack_msg:
                    print(f"解锁命令响应: command={ack_msg.command}, result={ack_msg.result}")
                self.mav_connection.mav.command_long_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 0, 0, 0, 0, 0, 0
                )
        if not armed:
            print("解锁失败")
            return False
        return True

    def wait_for_ekf_ready(self):
        """等待 EKF 完全收敛"""
        print("等待 EKF 收敛...")
        while True:
            msg = self.mav_connection.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=1)
            if msg:
                flags = msg.flags
                # 检查所有必要的 EKF 标志
                if (flags & mavutil.mavlink.EKF_ATTITUDE and
                        flags & mavutil.mavlink.EKF_VELOCITY_HORIZ and
                        flags & mavutil.mavlink.EKF_VELOCITY_VERT and
                        flags & mavutil.mavlink.EKF_POS_HORIZ_REL and
                        flags & mavutil.mavlink.EKF_POS_HORIZ_ABS and
                        flags & mavutil.mavlink.EKF_POS_VERT_ABS):
                    print("EKF 已收敛")
                    return True
            time.sleep(0.1)

    def wait_for_height(self,alt=100):
        """
        等待达到指定高度
        alt: 目标高度 (米)
        """
        print(f"等待达到目标高度: {alt} 米")
        start_time = time.time()
        while time.time() - start_time < 100:
            self.refresh_msg()
            if self.global_position_alt-self.alt_origin >= alt:
                self.logger.log(f"已达到目标高度: {alt} 米")
                break
    def wait_for_channel(self,channel=6):
        value=1500
        print(f"等待通道 {channel} 达到值 {value}")
        attr_name = f'chan{channel}_raw'
        while True:
            while True:
                msg=self.mav_connection.recv_msg()
                if msg is None:
                    break
            rc_msg = self.mav_connection.messages.get('RC_CHANNELS')
            if rc_msg is None:
                continue
            channel_value = getattr(rc_msg, attr_name, None)
            if channel_value is None:
                print(f"通道 {channel} 不存在对应的属性 {attr_name}")
                return False

            if channel_value >= value:
                print(f"通道 {channel} 已达到值 {value}，开始后续主进程")
                break
 

    def takeoff(self, alt):
        """
        takeoff命令只有在多旋翼无人机上起作用，固定翼无人机需手动使用油门起飞
        固定翼飞机使用takeoffwithoutGPS起飞
        """
        self.set_mode("GUIDED")
        self.set_mode("AUTO")

        # 2. 添加起飞任务
        self.mav_connection.mav.mission_item_send(
            self.mav_connection.target_system,
            self.mav_connection.target_component,
            0,  # 序号
            0,  # 当前航点帧
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # 当前航点
            1,  # 自动继续
            15,  # 起飞角度（通常10-15度）
            0, 0, 0,  # 未使用
            0, 0, alt  # 目标高度
        )

        print(f"开始起飞到 {alt} 米")

        start_time = time.time()
        reached_alt = False
        print("等待达到目标高度...")

        while time.time() - start_time < 100:  # 60秒超时
            msg = self.mav_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                current_alt = msg.relative_alt / 1000.0  # 转换为米
                if abs(current_alt - alt) <= 1.0:  # 在1米误差范围内
                    reached_alt = True
                    print(f"已达到目标高度: {alt}米")
                    break

        if not reached_alt:
            print("未能在规定时间内达到目标高度")
            return False

    def set_local_position(self, x, y, z):
        """
        x: 北向位移(米)
        y: 东向位移(米)
        z: 向下位移(米，通常为负值)
        """
        self.set_mode("GUIDED")
        self.mav_connection.mav.set_position_target_local_ned_send(
            0,  # 时间戳
            self.mav_connection.target_system,
            self.mav_connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b110111111000,  # type_mask
            x, y, z,  # 位置
            0, 0, 0,  # 速度
            0, 0, 0,  # 加速度
            0, 0  # yaw, yaw_rate
        )

    def calculate_velocity(self, lat1, lon1, alt1, lat2, lon2, alt2, time_diff):
        """
        计算速度信息 (vn, ve, vd)
        lat1, lon1, alt1: 上一点坐标
        lat2, lon2, alt2: 当前点坐标
        time_diff: 时间间隔 (秒)
        """
        if time_diff == 0:
            return 0, 0, 0  # 避免除零错误

        # 转换经纬度到弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # 计算北向速度 vn (纬度方向)
        vn = (lat2 - lat1) * self.EARTH_RADIUS / time_diff

        # 计算东向速度 ve (经度方向, 需要乘以 cos(纬度))
        ve = (lon2 - lon1) * self.EARTH_RADIUS * math.cos((lat1 + lat2) / 2) / time_diff

        # 计算垂直速度 vd (高度方向)
        vd = (alt2 - alt1) / time_diff

        return vn, ve, vd

    def send_gps_input(self, lat, lon, alt):
        """
        发送GPS位置信息
        lat: 当前纬度
        lon: 当前经度
        alt: 当前高度 (米)
        暂时不使用和计算速度信息，位置信息的误差容许值调的较宽泛
        """
        # 记录时间
        current_time = int(time.monotonic() * 1e6)
        # 发送 MAVLink GPS_INPUT 消息
        if self.args['gps_send']=="nmea" and self.gps_serial!=None:
            gpgga=self.create_gpgga(lat,lon,alt)
            print(gpgga)
            self.gps_serial.write((gpgga + '\r\n').encode('ascii'))        
            gprmc=self.create_gprmc(lat,lon,self.vn,self.ve)
            print(gprmc)
            self.gps_serial.write((gprmc + '\r\n').encode('ascii'))

        elif self.args['gps_send']=="mav_vision_position_estimeate":
            self.mav_connection.mav.vision_position_estimate_send(
                int(current_time * 1e6),  # 微秒时间戳
                100, 100, 100,            # 视觉定位系统给出的坐标（米）
                1, 1, 1  # 姿态（弧度）
            )
        elif self.args['gps_send']=="mav_global_vision_position_estimate ":
            self.mav_connection.mav.global_vision_position_estimate_send(
                int(current_time * 1e6),  # 微秒时间戳
                100, 100, 100,            # 视觉定位系统给出的坐标（米）
                1, 1, 1  # 姿态（弧度）
            )

        elif self.args['gps_send']=="mav_gps_input":
            self.mav_connection.mav.gps_input_send(
                int(current_time * 1e6),  # 时间戳（微秒）
                1,  # gps_id
                0b00000000,  # ignore_flags (仅忽略 VD 速度)
                0,  # time_week_ms
                0,  # time_week
                3,  # fix_type (3D fix)
                int(lat * 1e7),  # lat - 纬度(度 * 1e7)
                int(lon * 1e7),  # lon - 经度(度 * 1e7)
                alt,  # alt - 高度(米)
                2.0,  # hdop - 水平精度因子
                2.0,  # vdop - 垂直精度因子
                self.vn,  # vn - 北向速度
                self.ve,  # ve - 东向速度
                self.vd,  # vd - 垂直速度
                3.0,  # speed_accuracy
                3.0,  # horiz_accuracy
                3.0,  # vert_accuracy
                8,  # satellites_visible
                0   # yaw
            )
    def create_gpgga(self,lat, lon, alt):
        # 获取当前时间，精确到毫秒
        now = datetime.now()
        time_str = now.strftime("%H%M%S.%f")[:-3]  # 格式化时间，保留3位毫秒
        
        # 确定半球方向
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        # 取绝对值并转换格式
        lat_nmea = format_nmea_degrees(abs(lat))
        lon_nmea = format_nmea_degrees(abs(lon))
        
        # 构建GPGGA语句（不含校验和）
        nmea_base = f"GPGGA,{time_str},{lat_nmea},{lat_dir},{lon_nmea},{lon_dir},1,08,01.2,{alt:.1f},M,0.0,M,,"
        
        # 计算校验和
        checksum = calculate_checksum("$" + nmea_base)
        
        # 完整的GPGGA语句
        gpgga = f"${nmea_base}*{checksum}"
        
        return gpgga

    def create_gprmc(self,lat, lon, vx, vy):
        """
        创建GPRMC NMEA语句
        
        参数:
        lat: 纬度 (十进制度数)
        lon: 经度 (十进制度数)
        vx: X轴速度 (m/s) - 北向为正
        vy: Y轴速度 (m/s) - 东向为正
        vz: Z轴速度 (m/s) - 可选，在GPRMC中不使用
        """
        # 获取当前时间，精确到毫秒
        now = datetime.now()
        time_str = now.strftime("%H%M%S.%f")[:-3]  # 格式化时间，保留3位毫秒
        date_str = now.strftime("%d%m%y")  # 日期格式: DDMMYY
        
        # 确定半球方向
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        # 取绝对值并转换格式
        lat_nmea = format_nmea_degrees(abs(lat))
        lon_nmea = format_nmea_degrees(abs(lon))
        
        # 计算地速 (speed over ground) - 从vx和vy计算
        # 转换为节 (knots): 1 m/s = 1.94384 knots
        speed_knots = math.sqrt(vx**2 + vy**2) * 1.94384
        
        # 计算地面航向 (course over ground) - 从vx和vy计算
        # 航向角: 0°为正北，顺时针增加
        cog=self.mav_connection.messages['GPS_RAW_INT'].cog / 100
        # cog2 = calculate_bearing(self.last_lat, self.last_lon, lat, lon)
        # GPRMC状态: A=有效定位，V=无效定位
        status = 'A'  # 假设我们有有效定位'
        
        # 构建GPRMC语句（不含校验和）
        
        nmea_base = (
            f"GPRMC,{time_str},{status},{lat_nmea},{lat_dir},{lon_nmea},{lon_dir},"
            f"{speed_knots:.2f},{cog:.2f},{date_str},,"
        )

        
        # 计算校验和
        checksum = calculate_checksum("$" + nmea_base)
        
        # 完整的GPRMC语句
        gprmc = f"${nmea_base}*{checksum}"
        
        return gprmc

    def send_guided_change_heading(self, heading_type, target_heading, heading_rate):
        """
        可以实现在全程无GPS情况下，发送改变航向的控制命令
        heading_type: 航向类型 (0: course-over-ground, 1: raw vehicle heading)
        target_heading: 目标航向(度, 0-359.99)
        heading_rate: 改变航向的速率(米/秒/秒)
        """
        self.set_mode("GUIDED")
        self.mav_connection.mav.command_long_send(
            self.mav_connection.target_system,  # target system
            self.mav_connection.target_component,  # target component
            mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_HEADING,  # command (MAV_CMD_GUIDED_CHANGE_HEADING)
            0,  # confirmation
            heading_type,  # param1: 航向类型
            target_heading,  # param2: 目标航向(度)
            heading_rate,  # param3: 航向改变速率
            0,  # param4: (空)
            0,  # param5: (空)
            0,  # param6: (空)
            0  # param7: (空)
        )

    def send_guided_waypoint(self, lat, lon, alt_relative):
        """
        指定一个全球坐标点，让固定翼飞机在GUIDED模式下飞向该点
        单点目标飞行，与mission航线任务飞行对应

        参数:
        lat: 纬度
        lon: 经度
        alt_relative: 相对起飞点的高度(米)
        """
        # 首先切换到GUIDED模式
        self.set_mode("GUIDED")

        # 发送MISSION_ITEM消息
        self.mav_connection.mav.mission_item_send(
            self.mav_connection.target_system,  # target system
            self.mav_connection.target_component,  # target component
            0,  # sequence number (0)
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # frame
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,  # command (WAYPOINT)
            2,  # current (2 = guided mode waypoint)
            1,  # autocontinue
            0, 0, 0, 0,  # param1-4 (未使用)
            lat, lon, alt_relative  # x(lat), y(lon), z(alt)
        )

        # 等待命令被接受
        msg = self.mav_connection.recv_match(type=['MISSION_ACK'], timeout=3)
        if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
            print("Waypoint accepted")
            return True
        else:
            print("Waypoint not accepted")
            return False

    def takeoff_without_gps(self):
        """无GPS条件下的固定翼起飞"""
        # 确保在 STABILIZE 模式
        self.set_mode("FBWA")
        time.sleep(1)
        try:
            # 初始化所有通道为中位值
            # 1. 逐步增加油门
            self.rc_values[4] = 0
            self.logger.log(f"开始加速")
            for throttle in range(1200, 1800, 50):  # 从中位值逐步增加到80%油门
                self.rc_values[2] = throttle  # 通道3是油门
                self.mav_connection.mav.rc_channels_override_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)

            # 2. 等待速度建立
            self.logger.log(f"保持速度")
            for _ in range(30):  # 保持3秒
                self.mav_connection.mav.rc_channels_override_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    *self.rc_values

                )
                time.sleep(0.1)

            # 3. 抬升机头
            self.logger.log(f"抬升机头")
            self.rc_values[1] = 1700  # 抬升
            for _ in range(150):  # 保持3秒
                self.mav_connection.mav.rc_channels_override_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)
            # 4. 保持一段时间让飞机爬升
            # 5. 恢复平飞姿态
            self.logger.log(f"调整为平飞")
            for _ in range(30):  # 保持3秒
                self.rc_values[1] = 1500  # 恢复平值
                self.mav_connection.mav.rc_channels_override_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)
            # 确保模式稳定
            self.logger.log(f"起飞结束")
            #启动thread维持飞机油门

            # self.send_guided_change_heading(0, 90, 10)
            return True

        except Exception as e:
            print(f"起飞过程出错: {str(e)}")
            # 恢复所有通道到中位
            return False

    def start_thorottle(self):
        thread = threading.Thread(target=self.thorottle_thread)
        thread.daemon = True  # 设为守护线程，主线程结束时子线程也结束
        thread.start()

    def key_press_handler(self, event):
        """按键按下处理"""
        key = event.name
        print("press: ", key)
        if key == 'W':  # 抬头
            self.rc_values[1] += 10  # 通道2抬升
        elif key == 'S':  # 低头
            self.rc_values[1] -= 10  # 通道2下压
        elif key == 'A':  # 左转
            self.rc_values[0] -= 10  # 副翼向左
        elif key == 'D':  # 右转
            self.rc_values[0] += 10  # 副翼向右
        elif key == 'up':  # 油门加
            self.rc_values[2] += 10  # 通道3 油门增加
        elif key == 'down':  # 油门减
            self.rc_values[2] -= 10  # 通道3 油门减小
        elif key == 'left':  # 左偏航
            self.rc_values[3] -= 10  # 通道4 方向舵左
        elif key == 'right':  # 右偏航
            self.rc_values[3] += 10  # 通道4 方向舵右
        elif key == '+':  # 加号增大转弯角度
            self.turn_angle += 10
            print("present turn_angle:", self.turn_angle)
        elif key == '-':  # 减号减小转弯角度
            self.turn_angle -= 10
            print("present turn_angle:", self.turn_angle)
        elif key == '4':  # 小键盘 4在当前基础上向左转
            self.current_yaw -= self.turn_angle
            if self.current_yaw < 0:
                self.current_yaw += 360
            print("target yaw:", self.current_yaw)
            self.send_guided_change_heading(1, self.current_yaw, self.turn_rate)
        elif key == '6':  # 小键盘 6 在当前基础上向右转
            self.current_yaw += self.turn_angle
            if self.current_yaw > 360:
                self.current_yaw -= 360
            print("target yaw:", self.current_yaw)
            self.send_guided_change_heading(1, self.current_yaw, self.turn_rate)
        elif key == '8':  # 数字键8加大转弯速率
            self.turn_rate += 5  # 通道1调整
            print("present turn_rate:", self.turn_rate)
        elif key == '2':  # 数字键2 减小转弯速率
            self.turn_rate -= 5  #
            if self.turn_rate < 0:
                self.turn_rate = 5
            print("present turn_rate:", self.turn_rate)

    def get_global_position(self):
        return self.real_lat, self.real_lon, self.press_alt, self.global_position_alt,self.global_position_lat, self.global_position_lon
    
    def refresh_msg(self, mode="fly_mode"):
        # while True:
        while True:
            msg=self.mav_connection.recv_msg()
            if msg is None:
                break
        if mode == 'sim':
            self.real_lat=self.mav_connection.messages['SIMSTATE'].lat / 1e7
            self.real_lon=self.mav_connection.messages['SIMSTATE'].lng / 1e7
        # elif msg_type == 'GLOBAL_POSITION_INT':
        self.global_position_lat=self.mav_connection.messages['GLOBAL_POSITION_INT'].lat / 1e7
        self.global_position_lon=self.mav_connection.messages['GLOBAL_POSITION_INT'].lon / 1e7
        self.global_position_alt=self.mav_connection.messages['GLOBAL_POSITION_INT'].alt/1000
        self.vn,self.ve,self.vd=self.mav_connection.messages['GLOBAL_POSITION_INT'].vx/100,self.mav_connection.messages['GLOBAL_POSITION_INT'].vy/100,self.mav_connection.messages['GLOBAL_POSITION_INT'].vz/100
        self.roll =self.mav_connection.messages['ATTITUDE'].roll
        self.pitch =self.mav_connection.messages['ATTITUDE'].pitch
        self.yaw =self.mav_connection.messages['ATTITUDE'].yaw
        press_abs=self.mav_connection.messages['SCALED_PRESSURE'].press_abs
        #根据压差计算高度
        self.press_alt= self.alt_origin + ((self.temp_origin+273.15) / 0.0065) * (1 - (press_abs / self.press_origin) ** ((287.05 * 0.0065 ) / 9.80665))

    def thorottle_thread(self):

        while True:          
            try:
                rc_channels=self.mav_connection.messages['RC_CHANNELS'].chan7_raw
                # 发送油门信号
                self.mav_connection.mav.rc_channels_override_send(
                    self.mav_connection.target_system,
                    self.mav_connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)  # 控制发送频率

            except Exception as e:
                print(f"[ERROR] 线程异常: {e}")
                time.sleep(0.1)  # 发生异常时稍作等待，避免死循环导致 CPU 过载

    def update_global_position(self, current_lat, current_lon, current_alt):
        """更新当前位置"""
        self.send_gps_input(current_lat, current_lon, current_alt)
        self.last_alt = current_alt
        self.last_lat = current_lat
        self.last_lon = current_lon

