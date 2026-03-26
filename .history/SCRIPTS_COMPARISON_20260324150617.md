# 脚本对比和选择指南

## 📋 三个脚本的功能对比

| 功能特性 | 基础脚本 | 高级脚本 | 部署脚本 |
|---------|---------|---------|---------|
| **文件名** | `campus_network_login.sh` | `campus_network_login_advanced.sh` | `install_campus_login.sh` |
| **文件大小** | ~3KB | ~10KB | ~8KB |
| **学习难度** | ⭐ 易 | ⭐⭐ 中 | ⭐⭐⭐ 难 |
| **功能完整度** | ⭐⭐ 基础 | ⭐⭐⭐⭐ 完整 | ⭐⭐⭐ 安装 |
| **自动配置** | ❌ 否 | ✅ 是 | ✅ 是 |
| **配置文件** | 无 | `campus_login.conf` | 自动生成 |
| **支持crontab** | ✅ 是 | ✅ 是 | ✅ 是 |
| **支持systemd** | ❌ 否 | ⚠️ 手动 | ✅ 自动 |
| **网络监控** | ✅ 基础 | ✅⭐ 实时 | - |
| **邮件告警** | ❌ 否 | ✅ 是 | ⚠️ 配置 |
| **日志管理** | ⭐ 简单 | ⭐⭐ 详细 | ✅ 自动轮转 |
| **命令行选项** | ❌ 否 | ✅ 多种 | - |
| **交互式配置** | ❌ 否 | ⭐ 部分 | ✅ 完全 |
| **彩色输出** | ❌ 否 | ✅ 是 | ✅ 是 |

---

## 🎯 选择建议

### 场景1：快速部署（推荐新手）
```
使用脚本：install_campus_login.sh
原因：
  ✅ 一键自动配置
  ✅ 自动安装依赖
  ✅ 交互式引导
  ✅ 自动测试登录
```

**部署命令：**
```bash
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh
```

---

### 场景2：已有配置，需要功能完整
```
使用脚本：campus_network_login_advanced.sh
原因：
  ✅ 功能最完整
  ✅ 支持多种操作模式
  ✅ 实时监控能力强
  ✅ 邮件告警支持
```

**使用命令：**
```bash
# 生成配置文件
/usr/local/bin/campus_network_login_advanced.sh --generate-config

# 编辑配置
sudo nano /etc/campus_login.conf

# 测试登录
/usr/local/bin/campus_network_login_advanced.sh --login

# 实时监控
/usr/local/bin/campus_network_login_advanced.sh --monitor
```

---

### 场景3：只需要基础功能，脚本最小化
```
使用脚本：campus_network_login.sh
原因：
  ✅ 脚本最小
  ✅ 依赖最少
  ✅ 易于理解和修改
  ✅ 内存占用最低
```

**设置步骤：**
```bash
# 1. 编辑脚本配置部分（第15-24行）
sudo nano campus_network_login.sh

# 2. 手动修改参数：
USERNAME="your_id"
PASSWORD="your_pwd"
AC_ID="1"

# 3. 测试
./campus_network_login.sh
```

---

## 🚀 快速开始（按场景分类）

### 新手用户（推荐）
```bash
# 第1步：自动部署
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh

# 脚本会自动：
# - 检查依赖
# - 安装脚本
# - 配置参数
# - 测试登录
# - 设置自启

# 完成！
```

### 高级用户
```bash
# 第1步：安装脚本
sudo cp campus_network_login_advanced.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/campus_network_login_advanced.sh

# 第2步：生成配置
sudo /usr/local/bin/campus_network_login_advanced.sh --generate-config

# 第3步：编辑配置
sudo nano /etc/campus_login.conf

# 第4步：测试
/usr/local/bin/campus_network_login_advanced.sh --login

# 第5步：启动监控（可选）
/usr/local/bin/campus_network_login_advanced.sh --monitor &
```

### 极简用户
```bash
# 第1步：编辑基础脚本
nano campus_network_login.sh
# 修改第15-24行的参数

# 第2步：测试
bash campus_network_login.sh

# 第3步：加入crontab
crontab -e
# 添加：*/5 * * * * /path/to/campus_network_login.sh
```

---

## 📊 脚本功能详解

### 基础脚本（campus_network_login.sh）

**核心功能：**
- 网络连通性检测
- 自动登录与重试
- 日志记录
- 容错处理

**典型用途：**
```bash
# 简单的crontab任务
*/5 * * * * /usr/local/bin/campus_network_login.sh
```

**优点：**
- 代码简洁（~150行）
- 易于修改和定制
- 依赖最少

**缺点：**
- 需要手动编辑脚本修改参数
- 无配置文件支持
- 命令行选项少

---

### 高级脚本（campus_network_login_advanced.sh）

**核心功能：**
- 基础脚本的所有功能
- 配置文件支持
- 多种操作模式
- 实时网络监控
- 邮件告警系统
- 彩色输出和状态显示

**典型用途：**
```bash
# 生成配置文件
/usr/local/bin/campus_network_login_advanced.sh --generate-config

# 执行一次性登录
/usr/local/bin/campus_network_login_advanced.sh --login

# 实时监控网络
/usr/local/bin/campus_network_login_advanced.sh --monitor

# 查看网络状态
/usr/local/bin/campus_network_login_advanced.sh --status

# 检查网络连通性
/usr/local/bin/campus_network_login_advanced.sh --check
```

**优点：**
- 功能最完整
- 配置文件管理
- 多种操作模式
- 邮件告警支持
- 详细的网络信息

**缺点：**
- 脚本较长（~350行）
- 依赖项相对多

---

### 部署脚本（install_campus_login.sh）

**核心功能：**
- 自动依赖检查和安装
- 脚本自动复制和权限设置
- 交互式配置向导
- 自动登录测试
- Crontab自启配置
- Systemd服务配置
- 日志轮转设置

**典型用途：**
```bash
# 一键部署（需要root权限）
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh

# 脚本会提示你：
# 1. 输入学号和密码
# 2. 选择登录URL
# 3. 选择执行频率
# 4. 选择是否配置systemd
# 5. 自动执行登录测试
```

**优点：**
- 完全自动化部署
- 交互式配置无需编辑文件
- 自动依赖管理
- 配置验证
- 开箱即用

**缺点：**
- 只能一次性配置
- 需要root权限

---

## 🔄 脚本之间的关系

```
install_campus_login.sh（部署脚本）
    ↓
    ├─→ 安装 campus_network_login.sh（基础脚本）
    │   └─→ 用于简单的crontab任务
    │
    └─→ 安装 campus_network_login_advanced.sh（高级脚本）
        └─→ 用于高级功能和监控
        └─→ 用于交互式命令
```

---

## 🛠️ 配置文件位置汇总

| 项目 | 位置 | 创建方式 |
|------|------|---------|
| 配置文件 | `/etc/campus_login.conf` | 部署脚本自动或高级脚本生成 |
| 日志文件 | `/var/log/campus_login.log` | 脚本首次运行自动创建 |
| 脚本文件 | `/usr/local/bin/campus_network_login.sh` | 部署脚本自动复制 |
| 高级脚本 | `/usr/local/bin/campus_network_login_advanced.sh` | 部署脚本自动复制 |
| Crontab | 系统crontab | 部署脚本或手动配置 |
| Systemd服务 | `/etc/systemd/system/campus-login.service` | 部署脚本或高级脚本 |
| Systemd定时器 | `/etc/systemd/system/campus-login.timer` | 部署脚本或高级脚本 |
| Logrotate | `/etc/logrotate.d/campus-login` | 部署脚本自动配置 |

---

## 🔐 安全性考虑

### 密码存储

所有脚本都支持以下安全方式：

**方式1：配置文件（权限600）**
```bash
# 配置文件只有root可读
sudo chmod 600 /etc/campus_login.conf
```

**方式2：环境变量**
```bash
export CAMPUS_USERNAME="your_id"
export CAMPUS_PASSWORD="your_pwd"
```

**方式3：密钥存储（高级）**
```bash
# 使用系统密钥管理工具
sudo apt install pass
pass insert campus/password
```

---

## 📈 性能对比

| 指标 | 基础脚本 | 高级脚本 | 部署脚本 |
|------|----------|----------|----------|
| 启动时间 | ~0.5s | ~0.7s | N/A |
| 运行时间 | ~1-3s | ~1-3s | ~2分钟 |
| 内存占用 | ~3MB | ~5MB | ~10MB |
| CPU占用 | 极低 | 极低 | 中等 |
| 磁盘占用 | ~3KB | ~10KB | ~8KB |

---

## 🎓 学习路径建议

### 初学者
```
1. 阅读 QUICK_START.md
2. 运行部署脚本 install_campus_login.sh
3. 查看日志验证运行
4. 查看 CAMPUS_LOGIN_GUIDE.md 了解更多
```

### 中级用户
```
1. 使用高级脚本 campus_network_login_advanced.sh
2. 学习配置文件管理
3. 尝试不同的监控和诊断命令
4. 配置邮件告警
```

### 高级用户
```
1. 阅读基础脚本源码
2. 修改和定制脚本
3. 集成其他工具和系统服务
4. 开发自己的扩展功能
```

---

## 📞 故障排查导航

| 问题 | 推荐查阅 |
|------|---------|
| 不知道选哪个脚本 | 本文件（脚本对比和选择指南） |
| 部署过程出错 | QUICK_START.md 的"常见问题和解决方案" |
| 配置参数不对 | CAMPUS_LOGIN_GUIDE.md 的"获取认证参数" |
| 登录总是失败 | CAMPUS_LOGIN_GUIDE.md 的"故障排查" |
| 想看日志 | CAMPUS_LOGIN_GUIDE.md 的"日志管理" |
| 需要高级功能 | 使用 `--help` 或查看高级脚本源码 |

---

## 📝 推荐配置清单

### 最小化配置（仅需基础脚本）
- [ ] 编辑基础脚本参数
- [ ] 测试登录功能
- [ ] 加入crontab

### 标准配置（推荐新手）
- [ ] 运行部署脚本
- [ ] 验证日志输出
- [ ] 测试crontab自启

### 完整配置（推荐高级用户）
- [ ] 安装高级脚本
- [ ] 生成配置文件
- [ ] 配置systemd定时器
- [ ] 启用邮件告警
- [ ] 设置日志轮转
- [ ] 启动监控模式

---

祝你使用愉快！🎉

如有其他问题，请参考其他文档：
- [快速开始指南](QUICK_START.md)
- [详细使用指南](CAMPUS_LOGIN_GUIDE.md)
