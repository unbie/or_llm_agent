#!/bin/bash

################################################################################
# 校园网自动登录脚本 - 高级版本
# 功能增强：
# - 支持配置文件
# - 网络监控和自动恢复
# - 电子邮件告警（可选）
# - 详细的性能统计
################################################################################

# 配置文件路径
CONFIG_FILE="${CONFIG_FILE:-/etc/campus_login.conf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认配置值
LOGIN_URL="${LOGIN_URL:-http://172.18.3.3}"
TEST_HOST="${TEST_HOST:-172.18.3.3}"
USERNAME="${USERNAME:-}"
PASSWORD="${PASSWORD:-}"
AC_ID="${AC_ID:-1}"
LOG_FILE="${LOG_FILE:-/var/log/campus_login.log}"
RETRY_INTERVAL="${RETRY_INTERVAL:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
ENABLE_EMAIL_ALERT="${ENABLE_EMAIL_ALERT:-false}"
EMAIL_TO="${EMAIL_TO:-}"

# ========== 加载配置文件 ==========
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

# 验证必要的参数
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "错误: 未设置USERNAME或PASSWORD，请配置 $CONFIG_FILE"
    exit 1
fi

# ========== 函数定义 ==========

# 彩色输出函数
print_color() {
    local color=$1
    local message=$2
    case $color in
        red)    echo -e "\033[31m${message}\033[0m" ;;
        green)  echo -e "\033[32m${message}\033[0m" ;;
        yellow) echo -e "\033[33m${message}\033[0m" ;;
        blue)   echo -e "\033[34m${message}\033[0m" ;;
        *)      echo "$message" ;;
    esac
}

# 带时间戳的日志记录
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 确保日志目录存在
    local log_dir=$(dirname "$LOG_FILE")
    mkdir -p "$log_dir" 2>/dev/null
    
    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

# 发送警告邮件（可选）
send_alert_email() {
    local subject="$1"
    local content="$2"
    
    if [ "$ENABLE_EMAIL_ALERT" != "true" ] || [ -z "$EMAIL_TO" ]; then
        return
    fi
    
    if command -v mail &> /dev/null; then
        echo "$content" | mail -s "$subject" "$EMAIL_TO"
        log_message "INFO" "邮件告警已发送: $subject"
    elif command -v sendmail &> /dev/null; then
        {
            echo "Subject: $subject"
            echo ""
            echo "$content"
        } | sendmail "$EMAIL_TO"
        log_message "INFO" "邮件告警已发送: $subject"
    fi
}

# 获取网络统计信息
get_network_stats() {
    local stats=""
    
    # 获取IP地址
    local ip=$(hostname -I | awk '{print $1}')
    [ -n "$ip" ] && stats="${stats}IP: $ip\n"
    
    # 获取网络接口状态
    local interfaces=$(ip link show | grep "state UP" | awk '{print $2}' | tr -d ':' | tr '\n' ',')
    [ -n "$interfaces" ] && stats="${stats}活跃接口: $interfaces\n"
    
    # 获取DNS信息
    local dns=$(cat /etc/resolv.conf 2>/dev/null | grep nameserver | awk '{print $2}' | tr '\n' ',')
    [ -n "$dns" ] && stats="${stats}DNS服务器: $dns\n"
    
    echo -e "$stats"
}

# 检查网络连接
check_network() {
    # 方法1: ping检查
    if ping -c 1 -W 2 "$TEST_HOST" &> /dev/null; then
        return 0
    fi
    
    # 方法2: curl检查
    local http_code=$(curl -m 3 -s -o /dev/null -w "%{http_code}" "$LOGIN_URL" 2>/dev/null)
    if [ "$http_code" != "000" ]; then
        return 0
    fi
    
    return 1
}

# 执行登录
perform_login() {
    local attempt=1
    local login_start_time=$(date +%s)
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log_message "INFO" "第 $attempt 次登录尝试..."
        
        # 执行curl POST请求
        local response=$(curl -X POST \
            -d "username=${USERNAME}&password=${PASSWORD}&ac_id=${AC_ID}" \
            -m 5 \
            -w "\n%{http_code}" \
            -s \
            "$LOGIN_URL" 2>/dev/null)
        
        local http_code=$(echo "$response" | tail -n 1)
        local body=$(echo "$response" | head -n -1)
        
        log_message "INFO" "HTTP响应码: ${http_code}"
        
        # 判断登录是否成功
        if [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
            local login_duration=$(($(date +%s) - login_start_time))
            log_message "SUCCESS" "登录成功! (耗时: ${login_duration}秒)"
            
            # 发送成功告警
            send_alert_email \
                "校园网自动登录成功" \
                "时间: $(date)\n耗时: ${login_duration}秒\n网络信息:\n$(get_network_stats)"
            
            return 0
        fi
        
        # 检查是否已经在线
        if echo "$body" | grep -qi "already.*online\|already.*login\|已.*登录"; then
            log_message "SUCCESS" "已在线，无需重复登录"
            return 0
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            log_message "WARN" "登录失败 (响应码: $http_code)，${RETRY_INTERVAL}秒后重试..."
            sleep "$RETRY_INTERVAL"
        else
            log_message "ERROR" "登录失败，已尝试 $MAX_RETRIES 次"
            send_alert_email \
                "校园网自动登录失败告警" \
                "无法在 $MAX_RETRIES 次尝试后连接校园网\n最后响应码: $http_code\n网络信息:\n$(get_network_stats)"
        fi
        
        attempt=$((attempt + 1))
    done
    
    return 1
}

# 监控网络连接
monitor_network() {
    local check_count=0
    local last_status="unknown"
    
    log_message "INFO" "启动网络监控模式 (按Ctrl+C退出)..."
    
    while true; do
        check_count=$((check_count + 1))
        local check_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        if check_network; then
            if [ "$last_status" != "online" ]; then
                log_message "SUCCESS" "[$check_count] 网络在线"
                print_color "green" "✓ [$check_time] 网络在线"
                last_status="online"
            fi
        else
            if [ "$last_status" != "offline" ]; then
                log_message "WARN" "[$check_count] 检测到网络离线，开始登录..."
                print_color "yellow" "⚠ [$check_time] 网络离线，尝试登录..."
                perform_login
                last_status="offline"
            fi
        fi
        
        sleep 5  # 每5秒检查一次
    done
}

# 显示帮助信息
show_help() {
    cat << EOF
校园网自动登录脚本 - 高级版本

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -c, --check             检查网络连接状态
  -l, --login             执行登录（一次性）
  -m, --monitor           启动网络监控模式
  -s, --status            显示当前网络和登录状态
  -g, --generate-config   生成配置文件模板
  --username USERNAME     设置用户名（覆盖配置文件）
  --password PASSWORD     设置密码（覆盖配置文件）
  --url URL              设置登录URL（覆盖配置文件）

配置文件: $CONFIG_FILE

示例:
  # 执行一次性登录
  $0 --login

  # 启动监控模式
  $0 --monitor

  # 检查网络状态
  $0 --check

  # 使用命令行参数
  $0 --login --username=student_id --password=pwd

EOF
}

# 显示状态信息
show_status() {
    echo "======== 校园网登录脚本状态 ========"
    echo ""
    echo "配置信息:"
    echo "  登录URL: $LOGIN_URL"
    echo "  测试主机: $TEST_HOST"
    echo "  日志文件: $LOG_FILE"
    echo "  重试间隔: ${RETRY_INTERVAL}秒"
    echo "  最大重试次数: $MAX_RETRIES"
    echo ""
    echo "网络状态:"
    if check_network; then
        print_color "green" "  ✓ 网络在线"
    else
        print_color "red" "  ✗ 网络离线"
    fi
    echo ""
    echo "网络详情:"
    get_network_stats
    echo ""
    echo "最近日志:"
    tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
}

# 生成配置文件模板
generate_config() {
    cat > "$CONFIG_FILE" << 'EOF'
#!/bin/bash
# 校园网自动登录脚本配置文件
# 此文件将被脚本自动加载

# ========== 必需配置 ==========

# 登录用户名（学号）
USERNAME="your_student_id"

# 登录密码
PASSWORD="your_password"

# ========== 登录接口配置 ==========

# 登录页面地址
LOGIN_URL="http://172.18.3.3"

# 测试网络连通性的目标
TEST_HOST="172.18.3.3"

# 认证ID（通常为1，某些校园网需要修改）
AC_ID="1"

# ========== 日志配置 ==========

# 日志文件路径
LOG_FILE="/var/log/campus_login.log"

# ========== 重试配置 ==========

# 登录失败后的重试间隔（秒）
RETRY_INTERVAL=10

# 最大重试次数
MAX_RETRIES=3

# ========== 告警配置（可选） ==========

# 是否启用邮件告警 (true/false)
ENABLE_EMAIL_ALERT=false

# 告警邮箱地址（启用告警时必需）
# EMAIL_TO="your_email@example.com"

EOF
    
    if [ -f "$CONFIG_FILE" ]; then
        chmod 600 "$CONFIG_FILE"
        print_color "green" "✓ 配置文件已生成: $CONFIG_FILE"
        print_color "yellow" "请编辑配置文件并填入实际的用户名和密码"
        echo "编辑命令: sudo nano $CONFIG_FILE"
    fi
}

# 主程序
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            ;;
        -c|--check)
            echo "检查网络连接..."
            if check_network; then
                print_color "green" "✓ 网络在线"
                exit 0
            else
                print_color "red" "✗ 网络离线"
                exit 1
            fi
            ;;
        -l|--login)
            log_message "INFO" "======== 开始登录程序 ========"
            if check_network; then
                log_message "SUCCESS" "网络已在线，无需登录"
                print_color "green" "网络已在线"
            else
                if perform_login; then
                    sleep 2
                    if check_network; then
                        print_color "green" "✓ 登录成功，网络已恢复"
                    else
                        print_color "yellow" "⚠ 登录完成但网络仍未恢复"
                    fi
                else
                    print_color "red" "✗ 登录失败"
                    exit 1
                fi
            fi
            ;;
        -m|--monitor)
            monitor_network
            ;;
        -s|--status)
            show_status
            ;;
        -g|--generate-config)
            generate_config
            ;;
        --username=*|--username)
            USERNAME="${1#--username=}"
            log_message "INFO" "用户名已设置"
            ;;
        --password=*|--password)
            PASSWORD="${1#--password=}"
            log_message "INFO" "密码已设置"
            ;;
        --url=*|--url)
            LOGIN_URL="${1#--url=}"
            log_message "INFO" "URL已设置"
            ;;
        *)
            if [ -z "$1" ]; then
                # 无参数时执行默认的登录检查
                log_message "INFO" "======== 自动登录检查 ========"
                if check_network; then
                    log_message "SUCCESS" "网络正常，无需登录"
                else
                    log_message "WARN" "检测到网络离线，执行登录..."
                    perform_login
                fi
            else
                print_color "red" "未知选项: $1"
                show_help
                exit 1
            fi
            ;;
    esac
}

# 解析命令行参数（支持多个参数）
while [ $# -gt 0 ]; do
    case $1 in
        --username=*|--password=*|--url=*)
            param="${1%=*}"
            value="${1#*=}"
            case $param in
                --username) USERNAME="$value" ;;
                --password) PASSWORD="$value" ;;
                --url) LOGIN_URL="$value" ;;
            esac
            shift
            ;;
        *)
            main "$1"
            exit $?
            ;;
    esac
done

# 无参数时执行默认操作
[ $# -eq 0 ] && main
