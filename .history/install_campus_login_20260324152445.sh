#!/bin/bash

################################################################################
# 校园网自动登录脚本 - 自动部署工具
# 功能：自动化安装、配置和启动登录脚本
################################################################################

set -e  # 任何错误都会退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc"
CONFIG_FILE="$CONFIG_DIR/campus_login.conf"
LOG_DIR="/var/log"
CRON_IDENTIFIER="campus-network-login"

# ========== 工具函数 ==========

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ 校园网自动登录脚本 - 部署工具      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "此脚本需要root权限运行"
        echo "请使用以下命令运行: sudo $0"
        exit 1
    fi
}

check_dependencies() {
    echo ""
    echo "检查依赖工具..."
    
    local missing_deps=()
    
    # 检查curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    else
        print_success "curl已安装"
    fi
    
    # 检查ping（通常预装）
    if ! command -v ping &> /dev/null; then
        missing_deps+=("iputils-ping")
    else
        print_success "ping已安装"
    fi
    
    # 检查cron
    if ! command -v crontab &> /dev/null; then
        missing_deps+=("cron")
    else
        print_success "cron已安装"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warning "检测到缺失的依赖: ${missing_deps[*]}"
        echo "正在安装..."
        apt-get update -qq
        apt-get install -y "${missing_deps[@]}" > /dev/null 2>&1
        print_success "依赖已安装"
    fi
}

install_script() {
    echo ""
    echo "安装登录脚本..."
    
    # 安装基础脚本
    if [ -f "$SCRIPT_DIR/campus_network_login.sh" ]; then
        cp "$SCRIPT_DIR/campus_network_login.sh" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/campus_network_login.sh"
        print_success "基础脚本已安装到 $INSTALL_DIR/campus_network_login.sh"
    else
        print_error "找不到 campus_network_login.sh"
        return 1
    fi
    
    # 安装高级脚本（如果存在）
    if [ -f "$SCRIPT_DIR/campus_network_login_advanced.sh" ]; then
        cp "$SCRIPT_DIR/campus_network_login_advanced.sh" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/campus_network_login_advanced.sh"
        print_success "高级脚本已安装到 $INSTALL_DIR/campus_network_login_advanced.sh"
    fi
    
    return 0
}

setup_config() {
    echo ""
    echo "配置脚本参数..."
    
    # 如果配置文件已存在，询问是否覆盖
    if [ -f "$CONFIG_FILE" ]; then
        read -p "配置文件已存在，是否覆盖? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "使用现有配置文件"
            return 0
        fi
    fi
    
    echo ""
    echo "请输入认证参数:"
    
    # 获取用户输入
    read -p "请输入学号/用户名: " username
    read -sp "请输入密码 (不会显示): " password
    echo
    read -p "请输入认证ID (默认为1): " ac_id
    ac_id=${ac_id:-1}
    
    read -p "请输入登录URL (默认: http://172.18.3.3): " login_url
    login_url=${login_url:-http://172.18.3.3}
    
    # 生成配置文件
    cat > "$CONFIG_FILE" << EOF
#!/bin/bash
# 校园网自动登录脚本配置文件
# 自动生成于: $(date)

# 登录用户名（学号）
USERNAME="$username"

# 登录密码
PASSWORD="$password"

# 登录页面地址
LOGIN_URL="$login_url"

# 测试网络连通性的目标
TEST_HOST="172.18.3.3"

# 认证ID
AC_ID="$ac_id"

# 日志文件路径
LOG_FILE="/var/log/campus_login.log"

# 重试间隔（秒）
RETRY_INTERVAL=10

# 最大重试次数
MAX_RETRIES=3

# 邮件告警（可选）
ENABLE_EMAIL_ALERT=false
EOF
    
    chmod 600 "$CONFIG_FILE"
    print_success "配置文件已生成: $CONFIG_FILE"
}

test_login() {
    echo ""
    echo "测试登录功能..."
    
    read -p "是否要进行登录测试? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过登录测试"
        return 0
    fi
    
    print_info "执行登录测试..."
    if "$INSTALL_DIR/campus_network_login.sh"; then
        print_success "登录测试通过"
    else
        print_warning "登录测试失败，请检查配置"
        echo "查看日志: tail -f /var/log/campus_login.log"
    fi
}

setup_cron() {
    echo ""
    echo "配置开机自启..."
    
    read -p "是否配置crontab开机自启? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过crontab配置"
        return 0
    fi
    
    echo ""
    echo "选择执行频率:"
    echo "1) 每分钟检查一次（推荐用于频繁掉线）"
    echo "2) 每5分钟检查一次（推荐）"
    echo "3) 每10分钟检查一次"
    echo "4) 开机时运行一次"
    echo "5) 自定义"
    
    read -p "请选择 (1-5): " choice
    
    local cron_expr=""
    case $choice in
        1)
            cron_expr="* * * * * $INSTALL_DIR/campus_network_login.sh"
            ;;
        2)
            cron_expr="*/5 * * * * $INSTALL_DIR/campus_network_login.sh"
            ;;
        3)
            cron_expr="*/10 * * * * $INSTALL_DIR/campus_network_login.sh"
            ;;
        4)
            cron_expr="@reboot sleep 60 && $INSTALL_DIR/campus_network_login.sh"
            ;;
        5)
            read -p "请输入crontab表达式: " cron_expr
            ;;
        *)
            print_error "无效选择"
            return 1
            ;;
    esac
    
    # 添加到crontab
    local crontab_content
    crontab_content=$(crontab -l 2>/dev/null | grep -v "^#.*$CRON_IDENTIFIER" || true)
    echo "$crontab_content" | crontab -
    
    (crontab -l 2>/dev/null || echo ""; echo "# $CRON_IDENTIFIER"; echo "$cron_expr") | crontab -
    
    print_success "Crontab已配置"
    echo "当前crontab任务:"
    crontab -l | grep "$CRON_IDENTIFIER" || echo "（无相关任务）"
}

setup_systemd() {
    echo ""
    echo "是否配置systemd服务? (y/n) "
    read -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过systemd配置"
        return 0
    fi
    
    # 创建服务文件
    cat > /etc/systemd/system/campus-login.service << 'EOF'
[Unit]
Description=Campus Network Auto-Login Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/campus_network_login.sh
StandardOutput=journal
StandardError=journal
EOF
    
    # 创建定时器
    cat > /etc/systemd/system/campus-login.timer << 'EOF'
[Unit]
Description=Campus Network Auto-Login Timer
Requires=campus-login.service

[Timer]
OnBootSec=60s
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    systemctl daemon-reload
    systemctl enable campus-login.timer
    systemctl start campus-login.timer
    
    print_success "Systemd服务已配置"
    systemctl status campus-login.timer
}

setup_logrotate() {
    echo ""
    echo "配置日志轮转..."
    
    cat > /etc/logrotate.d/campus-login << 'EOF'
/var/log/campus_login.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOF
    
    print_success "日志轮转已配置"
}

show_summary() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ 安装完成！                                  ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    echo "已安装的文件:"
    echo "  脚本: $INSTALL_DIR/campus_network_login.sh"
    echo "  配置: $CONFIG_FILE"
    echo "  日志: $LOG_DIR/campus_login.log"
    echo ""
    echo "常用命令:"
    echo "  # 手动执行登录"
    echo "  $INSTALL_DIR/campus_network_login.sh"
    echo ""
    echo "  # 查看日志"
    echo "  tail -f $LOG_DIR/campus_login.log"
    echo ""
    echo "  # 查看crontab"
    echo "  crontab -l"
    echo ""
    echo "  # 高级命令（使用高级脚本）"
    echo "  $INSTALL_DIR/campus_network_login_advanced.sh --help"
    echo ""
    echo "下一步:"
    print_warning "1. 在浏览器中访问 http://172.18.3.3 确认登录页面地址"
    print_warning "2. 检查配置文件 $CONFIG_FILE 中的参数是否正确"
    print_warning "3. 运行命令测试: $INSTALL_DIR/campus_network_login.sh"
    echo ""
}

# ========== 主程序 ==========

main() {
    print_header
    check_root
    check_dependencies
    install_script
    setup_config
    test_login
    setup_cron
    setup_systemd
    setup_logrotate
    show_summary
}

# 执行主程序
main
