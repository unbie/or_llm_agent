# 校园网自动登录脚本使用指南

## 📋 概述

这是一个为合肥工业大学校园网设计的自动登录脚本，可以在网络断开时自动检测并重连。

## 🔧 前置准备

### 1. 获取认证参数

需要从校园网登录页面获取以下信息：

#### 方法A：使用浏览器F12查看
1. 打开 http://172.18.3.3
2. 按 F12 打开开发者工具
3. 转到 Network 标签
4. 输入用户名和密码，点击登录
5. 找到 POST 请求，查看：
   - **Request URL**: 登录接口地址
   - **Form Data**: 包含参数名称（如 `username`, `password`, `ac_id` 等）

#### 方法B：查看页面源码
1. 右键点击登录页面 → "查看网页源代码"
2. 搜索 `form` 标签，找到：
   - `action` 属性：POST 目标地址
   - `input` 标签中的 `name` 属性：参数名

### 2. 典型参数示例

根据校园网不同版本，参数可能包括：

```bash
# 参数示例 1: 简单认证
username=your_student_id
password=your_password
ac_id=1

# 参数示例 2: 扩展认证（某些版本需要）
username=your_student_id
password=your_password
ac_id=1
save_me=1
action=login
```

## ⚙️ 脚本配置步骤

### 1. 编辑脚本文件

```bash
sudo nano campus_network_login.sh
```

### 2. 修改以下参数（第15-24行）

```bash
USERNAME="your_username"           # 替换为学号
PASSWORD="your_password"           # 替换为密码
AC_ID="1"                          # 替换为认证ID
LOGIN_URL="http://172.18.3.3"      # 登录接口地址
```

### 3. 验证curl POST命令

在脚本执行登录前，建议先手动测试：

```bash
# 测试登录命令（根据实际参数修改）
curl -X POST \
  -d "username=your_username&password=your_password&ac_id=1" \
  http://172.18.3.3
```

预期响应：
- ✅ 成功：HTTP 200 或 302 状态码
- ❌ 失败：其他错误码或"Already Online"信息

## 🚀 脚本使用

### 1. 设置执行权限

```bash
chmod +x campus_network_login.sh
```

### 2. 手动测试

```bash
./campus_network_login.sh
```

查看输出结果：

```bash
tail -f /var/log/campus_login.log
```

### 3. 日志示例

✅ 成功登录：
```
[2024-03-24 10:30:45] ======== 网络登录检测启动 ========
[2024-03-24 10:30:45] 检测到网络断开，启动登录程序...
[2024-03-24 10:30:45] 第 1 次登录尝试...
[2024-03-24 10:30:46] HTTP响应码: 200
[2024-03-24 10:30:46] 登录成功！响应: ...
```

❌ 登录失败：
```
[2024-03-24 10:30:45] 检测到网络断开，启动登录程序...
[2024-03-24 10:30:45] 第 1 次登录尝试...
[2024-03-24 10:30:46] HTTP响应码: 403
[2024-03-24 10:30:46] 登录失败，10秒后重试...
```

## 🔁 开机自启配置（crontab）

### 方案1：使用crontab（推荐）

#### 1. 将脚本复制到系统目录

```bash
sudo cp campus_network_login.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/campus_network_login.sh
```

#### 2. 编辑crontab

```bash
sudo crontab -e
```

#### 3. 添加以下行（选择一个）

**选项A：每分钟检查一次（高频，耗资源少）**
```cron
* * * * * /usr/local/bin/campus_network_login.sh
```

**选项B：每5分钟检查一次（推荐）**
```cron
*/5 * * * * /usr/local/bin/campus_network_login.sh
```

**选项C：开机后延迟1分钟运行，然后每10分钟检查一次**
```cron
@reboot sleep 60 && /usr/local/bin/campus_network_login.sh
*/10 * * * * /usr/local/bin/campus_network_login.sh
```

**选项D：仅在开机时运行**
```cron
@reboot /usr/local/bin/campus_network_login.sh
```

#### 4. 验证crontab是否生效

```bash
# 查看当前用户的crontab
sudo crontab -l

# 查看系统日志
sudo journalctl -u cron -f

# 查看脚本日志
tail -f /var/log/campus_login.log
```

### 方案2：使用systemd服务（更灵活）

如果系统使用systemd，可以创建服务+定时器：

#### 1. 创建服务文件

```bash
sudo nano /etc/systemd/system/campus-login.service
```

内容：
```ini
[Unit]
Description=Campus Network Auto-Login Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/campus_network_login.sh
StandardOutput=journal
StandardError=journal
```

#### 2. 创建定时器文件

```bash
sudo nano /etc/systemd/system/campus-login.timer
```

内容：
```ini
[Unit]
Description=Campus Network Auto-Login Timer
Requires=campus-login.service

[Timer]
# 开机后60秒运行
OnBootSec=60s
# 每5分钟运行一次
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

#### 3. 启用并启动定时器

```bash
sudo systemctl daemon-reload
sudo systemctl enable campus-login.timer
sudo systemctl start campus-login.timer

# 查看定时器状态
sudo systemctl status campus-login.timer
sudo systemctl list-timers campus-login.timer
```

## 📊 日志管理

### 1. 防止日志过大

脚本每次运行都会追加日志，建议设置日志轮转：

```bash
sudo nano /etc/logrotate.d/campus-login
```

添加内容：
```
/var/log/campus_login.log {
    daily              # 每天轮转一次
    rotate 7           # 保留7个备份
    compress           # 压缩旧日志
    delaycompress      # 延迟压缩到下次轮转
    missingok          # 文件不存在也不报错
    notifempty         # 空文件不轮转
    create 0644 root root  # 创建新文件的权限
}
```

### 2. 查看日志

```bash
# 实时查看日志
tail -f /var/log/campus_login.log

# 查看最后100行
tail -100 /var/log/campus_login.log

# 搜索失败记录
grep "失败" /var/log/campus_login.log

# 查看登录成功的时间
grep "成功" /var/log/campus_login.log
```

### 3. 清理日志

```bash
# 清空日志（保留文件）
echo "" | sudo tee /var/log/campus_login.log

# 删除日志
sudo rm /var/log/campus_login.log
```

## 🐛 故障排查

### 问题1：脚本无法执行权限

```bash
chmod +x campus_network_login.sh
```

### 问题2：找不到log文件或无权限写入

脚本会自动降级到 `/tmp/campus_login.log`，确保有权限。

### 问题3：登录总是失败

1. **验证网络连通性**
   ```bash
   ping -c 1 172.18.3.3
   curl -v http://172.18.3.3
   ```

2. **测试登录命令**
   ```bash
   curl -X POST -v \
     -d "username=your_username&password=your_password&ac_id=1" \
     http://172.18.3.3
   ```

3. **检查参数是否正确**
   - 用户名/学号是否正确
   - 密码是否正确（避免特殊字符问题）
   - ac_id 是否正确

4. **检查登录接口**
   - 打开 http://172.18.3.3 查看最新登录页面
   - 检查 F12 Network 中的 POST 请求

### 问题4：crontab没有执行

```bash
# 检查cron服务是否运行
systemctl status cron

# 重启cron服务
sudo systemctl restart cron

# 查看crontab记录
sudo tail -f /var/log/syslog | grep CRON
```

## 📝 高级配置

### 1. 多个登录接口支持

如果有多个登录接口，可修改脚本添加循环：

```bash
LOGIN_URLS=("http://172.18.3.3" "http://172.18.3.4")
for url in "${LOGIN_URLS[@]}"; do
    # 尝试登录...
done
```

### 2. 性能优化

- 增加 `RETRY_INTERVAL` 时间：减少重试频率
- 减少 `MAX_RETRIES`：更快失败
- 修改curl超时 `-m 5`：调整等待时间

### 3. 安全改进

```bash
# 使用环境变量避免密码硬编码
read -sp "请输入密码: " PASSWORD

# 或从配置文件读取（受权限保护）
source /etc/campus_login.conf  # chmod 600
```

## 📞 常见问题

**Q: 脚本会不会产生大量网络流量？**
A: 不会。每5分钟只发送一个简单的POST请求，流量极小。

**Q: 脚本可以关闭吗？**
A: 是的，通过以下命令：
```bash
# crontab方式
sudo crontab -e  # 注释或删除对应行

# systemd方式
sudo systemctl stop campus-login.timer
sudo systemctl disable campus-login.timer
```

**Q: 如果频繁掉线怎么办？**
A: 可以改为每分钟检查一次，或检查：
- 网线连接是否松动
- WiFi信号是否稳定
- 校园网是否有使用限制

## 🎯 总结

| 步骤 | 命令 |
|------|------|
| 1. 获取参数 | 从登录页面F12获取 |
| 2. 编辑脚本 | `nano campus_network_login.sh` |
| 3. 设权限 | `chmod +x campus_network_login.sh` |
| 4. 测试 | `./campus_network_login.sh` |
| 5. 开机启动 | 使用crontab或systemd |
| 6. 监控日志 | `tail -f /var/log/campus_login.log` |

祝使用愉快！🎉
