# 🌐 合肥工业大学校园网自动登录脚本

> 一个为Ubuntu系统设计的校园网自动登录解决方案，支持自动检测网络、自动登录、定时检查和邮件告警。

## ✨ 主要特性

- ✅ **自动检测网络** - 实时监控网络连接状态
- ✅ **自动登录** - 网络断开时自动执行登录
- ✅ **容错处理** - 登录失败自动重试（可配置次数和间隔）
- ✅ **后台运行** - 支持crontab和systemd定时执行
- ✅ **日志管理** - 详细的日志记录和自动轮转
- ✅ **零配置部署** - 自动部署脚本一键安装
- ✅ **邮件告警** - 支持登录失败邮件通知
- ✅ **实时监控** - 高级脚本提供实时网络监控
- ✅ **多平台支持** - 适配各种Linux发行版

## 📁 文件说明

### 脚本文件
| 文件 | 说明 | 适合人群 |
|------|------|---------|
| **campus_network_login.sh** | 基础登录脚本，核心功能 | 所有用户 |
| **campus_network_login_advanced.sh** | 高级脚本，功能完整 | 进阶用户 |
| **install_campus_login.sh** | 自动部署脚本 | 新手用户 |

### 文档文件
| 文件 | 内容 | 何时阅读 |
|------|------|---------|
| **QUICK_START.md** | 5分钟快速开始 | 第一次使用 |
| **CAMPUS_LOGIN_GUIDE.md** | 完整使用指南 | 需要详细信息 |
| **SCRIPTS_COMPARISON.md** | 脚本功能对比 | 选择脚本时 |
| **README.md** | 本文件，项目概览 | 了解项目 |

## 🚀 快速开始

### 最简单的方式（推荐新手）

```bash
# 1. 下载脚本到本地
cd /path/to/scripts

# 2. 运行部署脚本
chmod +x install_campus_login.sh
sudo ./install_campus_login.sh

# 脚本会自动：
# ✓ 检查依赖并安装
# ✓ 复制脚本到系统目录
# ✓ 交互式配置参数
# ✓ 测试登录功能
# ✓ 配置开机自启
```

### 5分钟快速部署

**第1步：获取认证参数**
- 打开浏览器访问 http://172.18.3.3
- 按F12打开开发者工具，找到登录的POST请求
- 记录：username（学号）、password（密码）、ac_id（认证ID）

**第2步：运行部署脚本**
```bash
sudo ./install_campus_login.sh
```

**第3步：按提示输入参数**
```
请输入学号/用户名: 1810301001
请输入密码: ••••••••••
请输入认证ID (默认为1): 1
请输入登录URL (默认: http://172.18.3.3): [Enter]
```

**完成！** 脚本会自动配置开机自启和日志管理。

## 📋 详细使用步骤

### 步骤1：准备认证参数

从校园网登录页面获取以下信息：

```
登录页面：http://172.18.3.3
用户名：your_student_id （学号）
密码：your_password
认证ID：1 （通常为1）
```

**获取参数的方法：**
1. 在浏览器中打开 http://172.18.3.3
2. 按 F12 打开开发者工具
3. 切换到 Network 标签
4. 输入用户名密码，点击登录
5. 查看POST请求的Form Data

### 步骤2：选择安装方式

#### 方式A：自动部署（推荐 ⭐⭐⭐）
```bash
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh
```

#### 方式B：手动安装
```bash
# 复制脚本
sudo cp campus_network_login.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/campus_network_login.sh

# 编辑配置
sudo nano /usr/local/bin/campus_network_login.sh
# 修改第15-24行的参数

# 测试
/usr/local/bin/campus_network_login.sh

# 配置crontab
sudo crontab -e
# 添加：*/5 * * * * /usr/local/bin/campus_network_login.sh
```

### 步骤3：验证安装

```bash
# 查看脚本
ls -la /usr/local/bin/campus_network_login*

# 查看日志
tail -f /var/log/campus_login.log

# 查看crontab
sudo crontab -l
```

## 🔧 常用命令

### 基础操作

```bash
# 手动执行登录
/usr/local/bin/campus_network_login.sh

# 查看日志
tail -f /var/log/campus_login.log

# 查看crontab任务
sudo crontab -l

# 编辑crontab
sudo crontab -e
```

### 高级脚本命令（如果安装了高级脚本）

```bash
# 检查网络状态
/usr/local/bin/campus_network_login_advanced.sh --check

# 执行一次登录
/usr/local/bin/campus_network_login_advanced.sh --login

# 实时监控网络
/usr/local/bin/campus_network_login_advanced.sh --monitor

# 显示详细状态
/usr/local/bin/campus_network_login_advanced.sh --status

# 生成配置文件
/usr/local/bin/campus_network_login_advanced.sh --generate-config

# 查看帮助
/usr/local/bin/campus_network_login_advanced.sh --help
```

### 日志管理

```bash
# 实时查看日志
tail -f /var/log/campus_login.log

# 查看最近100行日志
tail -100 /var/log/campus_login.log

# 搜索成功记录
grep "成功" /var/log/campus_login.log

# 搜索失败记录
grep "失败" /var/log/campus_login.log

# 清空日志
echo "" | sudo tee /var/log/campus_login.log
```

## ⚙️ 配置说明

### 基本参数

```bash
# 学号/用户名
USERNAME="1810301001"

# 密码
PASSWORD="your_password"

# 认证ID（通常为1）
AC_ID="1"

# 登录URL
LOGIN_URL="http://172.18.3.3"
```

### 高级参数

```bash
# 测试网络的目标
TEST_HOST="172.18.3.3"

# 日志文件位置
LOG_FILE="/var/log/campus_login.log"

# 失败后的重试等待时间（秒）
RETRY_INTERVAL=10

# 最大重试次数
MAX_RETRIES=3

# 启用邮件告警
ENABLE_EMAIL_ALERT=false

# 告警邮箱（启用邮件告警时需要）
# EMAIL_TO="your_email@example.com"
```

## 📊 配置方案

### 方案1：频繁掉线（每分钟检查）
```bash
RETRY_INTERVAL=5
MAX_RETRIES=5

# crontab配置
* * * * * /usr/local/bin/campus_network_login.sh
```

### 方案2：稳定网络（每5分钟检查）
```bash
RETRY_INTERVAL=10
MAX_RETRIES=3

# crontab配置（推荐）
*/5 * * * * /usr/local/bin/campus_network_login.sh
```

### 方案3：仅开机自启
```bash
# crontab配置
@reboot sleep 60 && /usr/local/bin/campus_network_login.sh
```

### 方案4：每10分钟检查
```bash
# crontab配置
*/10 * * * * /usr/local/bin/campus_network_login.sh
```

## 🐛 常见问题

### Q1: 如何获取认证参数？
**A:** 
1. 打开 http://172.18.3.3
2. F12 打开开发者工具 → Network 标签
3. 输入用户名密码，点击登录
4. 查看 POST 请求的 Form Data

详见：[CAMPUS_LOGIN_GUIDE.md](CAMPUS_LOGIN_GUIDE.md#获取认证参数)

### Q2: 脚本可以关闭吗？
**A:** 是的，通过以下方式：

```bash
# 方式1：编辑crontab
sudo crontab -e
# 注释或删除相关行

# 方式2：删除systemd定时器
sudo systemctl disable campus-login.timer
sudo systemctl stop campus-login.timer
```

### Q3: 日志文件会变得很大吗？
**A:** 不会，自动部署脚本会配置日志轮转。每天自动压缩一次，保留7个备份。

### Q4: 登录总是失败怎么办？
**A:** 按以下步骤排查：

1. **验证网络**
   ```bash
   ping -c 1 172.18.3.3
   ```

2. **测试登录命令**
   ```bash
   curl -X POST -d "username=your_id&password=your_pwd&ac_id=1" http://172.18.3.3
   ```

3. **检查参数是否正确**
   - 用户名/学号
   - 密码（避免特殊字符）
   - ac_id

4. **查看详细日志**
   ```bash
   tail -50 /var/log/campus_login.log
   ```

详见：[CAMPUS_LOGIN_GUIDE.md 故障排查](CAMPUS_LOGIN_GUIDE.md#故障排查)

### Q5: 如何在远程服务器上使用？
**A:** 如果设备已配置Tailscale：

```bash
# 通过Tailscale连接到服务器
ssh user@server-ip

# 查看远程日志
tail -f /var/log/campus_login.log

# 手动触发登录
/usr/local/bin/campus_network_login.sh
```

更多问题请查看：[CAMPUS_LOGIN_GUIDE.md](CAMPUS_LOGIN_GUIDE.md)

## 📚 文档导航

| 需求 | 推荐文档 |
|------|---------|
| 快速开始 | [QUICK_START.md](QUICK_START.md) |
| 详细教程 | [CAMPUS_LOGIN_GUIDE.md](CAMPUS_LOGIN_GUIDE.md) |
| 脚本选择 | [SCRIPTS_COMPARISON.md](SCRIPTS_COMPARISON.md) |
| 故障排查 | [CAMPUS_LOGIN_GUIDE.md#故障排查](CAMPUS_LOGIN_GUIDE.md#故障排查) |

## 🎓 脚本对比

| 功能 | 基础脚本 | 高级脚本 | 部署脚本 |
|------|----------|----------|----------|
| 自动检测 | ✅ | ✅ | - |
| 自动登录 | ✅ | ✅ | - |
| 配置文件 | ❌ | ✅ | ✅ |
| 监控模式 | ❌ | ✅ | - |
| 邮件告警 | ❌ | ✅ | ⚠️ |
| 自动部署 | ❌ | ❌ | ✅ |
| crontab | ✅ | ✅ | ✅ |
| systemd | ❌ | ⚠️ | ✅ |

详见：[SCRIPTS_COMPARISON.md](SCRIPTS_COMPARISON.md)

## 📈 系统需求

- **操作系统**：Ubuntu 18.04 或更新
- **依赖工具**：
  - curl（用于HTTP请求）
  - ping（用于网络检测）
  - cron 或 systemd（用于定时执行）

### 自动检查和安装

部署脚本会自动检查和安装所有依赖：

```bash
sudo ./install_campus_login.sh
```

## 🔒 安全说明

### 密码存储

配置文件权限自动设置为 600（仅root可读）：

```bash
sudo ls -la /etc/campus_login.conf
# -rw------- 1 root root ...
```

### 其他安全建议

1. **定期修改密码** - 校园网密码定期更新时编辑配置文件
2. **限制文件权限** - 确保配置文件仅root可读
3. **定期检查日志** - 检查是否有异常登录记录
4. **备份配置** - 重要的配置文件定期备份

## 📞 技术支持

### 获取帮助

1. **查看脚本帮助**
   ```bash
   /usr/local/bin/campus_network_login_advanced.sh --help
   ```

2. **查看日志**
   ```bash
   tail -f /var/log/campus_login.log
   ```

3. **运行诊断**
   ```bash
   /usr/local/bin/campus_network_login_advanced.sh --status
   ```

4. **查看文档**
   - [QUICK_START.md](QUICK_START.md) - 快速开始
   - [CAMPUS_LOGIN_GUIDE.md](CAMPUS_LOGIN_GUIDE.md) - 详细指南
   - [SCRIPTS_COMPARISON.md](SCRIPTS_COMPARISON.md) - 功能对比

### 常见错误信息

| 错误 | 含义 | 解决方案 |
|------|------|---------|
| `HTTP响应码: 403` | 认证失败 | 检查用户名、密码、ac_id |
| `HTTP响应码: 000` | 网络无法连接 | 检查网络连接和登录URL |
| `已在线` | 无需登录 | 网络状态正常 |
| `登录失败，已尝试 3 次` | 重试次数已用尽 | 检查网络和参数配置 |

## 🌟 使用示例

### 示例1：基础使用
```bash
# 安装
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh

# 验证安装
/usr/local/bin/campus_network_login.sh

# 查看日志
tail -f /var/log/campus_login.log
```

### 示例2：高级监控
```bash
# 生成配置
/usr/local/bin/campus_network_login_advanced.sh --generate-config

# 编辑配置
sudo nano /etc/campus_login.conf

# 启动实时监控
/usr/local/bin/campus_network_login_advanced.sh --monitor
```

### 示例3：远程部署
```bash
# 在远程服务器上
ssh user@server-ip

# 下载脚本并安装
wget https://example.com/scripts.tar.gz
tar xzf scripts.tar.gz
sudo ./install_campus_login.sh

# 验证
tail -f /var/log/campus_login.log
```

## 📝 更新日志

### v1.1 (2024-03-24)
- ✅ 发布高级脚本版本
- ✅ 添加自动部署脚本
- ✅ 支持邮件告警
- ✅ 实时网络监控
- ✅ 详细的使用文档

### v1.0 (2024-03-24)
- ✅ 初始版本发布
- ✅ 基础网络检测和登录
- ✅ Crontab自启支持
- ✅ 日志记录功能

## 📄 许可证

MIT License - 自由使用和修改

## 🙏 感谢

感谢合肥工业大学网络运维部门的支持。

---

## 🎯 下一步

1. **快速开始** → 阅读 [QUICK_START.md](QUICK_START.md)
2. **详细了解** → 阅读 [CAMPUS_LOGIN_GUIDE.md](CAMPUS_LOGIN_GUIDE.md)
3. **选择脚本** → 参考 [SCRIPTS_COMPARISON.md](SCRIPTS_COMPARISON.md)
4. **开始部署** → 运行 `sudo ./install_campus_login.sh`

---

**祝你使用愉快！** 🎉

有问题？查看文档或检查 `/var/log/campus_login.log`。
