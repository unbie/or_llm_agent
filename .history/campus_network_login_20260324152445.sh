#!/bin/bash

################################################################################
# 合肥工业大学校园网自动登录脚本
# 功能：自动检测网络连接，在网络断开时自动登录
# 作者：Auto-Login Script
# 日期：2024
################################################################################

# ========== 配置部分 ==========
# 请根据实际情况修改以下参数

# 登录页面地址
LOGIN_URL="http://172.18.3.3"

# 测试网络连通性的目标地址（建议使用校园网内可访问的服务器）
TEST_URL="http://172.18.3.3"
TEST_HOST="172.18.3.3"

# curl POST 请求参数
# 请替换为实际的认证参数
# 常见参数包括: username、password、ac_id等
# 示例命令格式（需根据实际情况修改）：
# curl -X POST -d "username=YOUR_USERNAME&password=YOUR_PASSWORD&ac_id=YOUR_AC_ID" http://172.18.3.3

USERNAME="your_username"           # 替换为你的学号/用户名
PASSWORD="your_password"           # 替换为你的密码
AC_ID="1"                          # 替换为认证ID（通常可从登录页面源码找到）

# 日志文件路径
LOG_FILE="/var/log/campus_login.log"

# 重试间隔（秒）
RETRY_INTERVAL=10

# 最大重试次数
MAX_RETRIES=3

# ========== 函数定义 ==========

# 打印日志（带时间戳）
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" >> "${LOG_FILE}"
}

# 检查网络连接
check_network() {
    # 方法1: 使用ping检查网关连接
    if ping -c 1 -W 2 "${TEST_HOST}" &> /dev/null; then
        return 0  # 网络连通
    fi
    
    # 方法2: 使用curl检查HTTP连接（备用）
    if curl -m 3 -s "${TEST_URL}" &> /dev/null; then
        return 0
    fi
    
    return 1  # 网络不通
}

# 执行登录
perform_login() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log_message "第 $attempt 次登录尝试..."
        
        # 执行curl POST请求
        # 注意：需要根据实际的登录接口修改URL和参数
        local response=$(curl -X POST \
            -d "username=${USERNAME}&password=${PASSWORD}&ac_id=${AC_ID}" \
            -m 5 \
            -w "\n%{http_code}" \
            "${LOGIN_URL}" 2>/dev/null)
        
        local http_code=$(echo "$response" | tail -n 1)
        local body=$(echo "$response" | head -n -1)
        
        log_message "HTTP响应码: ${http_code}"
        
        # 根据响应码判断登录状态
        if [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
            log_message "登录成功！响应: $body"
            return 0
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            log_message "登录失败，${RETRY_INTERVAL}秒后重试..."
            sleep $RETRY_INTERVAL
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_message "登录失败，已尝试 $MAX_RETRIES 次"
    return 1
}

# 主程序
main() {
    log_message "======== 网络登录检测启动 ========"
    
    if check_network; then
        log_message "网络正常，无需登录"
        exit 0
    fi
    
    log_message "检测到网络断开，启动登录程序..."
    
    if perform_login; then
        log_message "自动登录完成，等待网络恢复..."
        # 等待一段时间后验证网络
        sleep 5
        if check_network; then
            log_message "网络已恢复！"
            exit 0
        else
            log_message "警告: 登录后网络仍未恢复"
            exit 1
        fi
    else
        log_message "自动登录失败，请检查脚本配置"
        exit 1
    fi
}

# ========== 脚本执行 ==========

# 确保日志文件可写
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE" 2>/dev/null || {
        LOG_FILE="/tmp/campus_login.log"
    }
fi

# 执行主程序
main
