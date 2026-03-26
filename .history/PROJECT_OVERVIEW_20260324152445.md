# 📦 校园网自动登录脚本 - 完整包说明

## 📋 项目概览

本项目为合肥工业大学宣城校区提供一套完整的校园网自动登录解决方案，包含三个功能层次的脚本和四份详细文档。

## 📂 文件清单

### 🔧 脚本文件（3个）

#### 1️⃣ `campus_network_login.sh` - 基础登录脚本

**用途**：核心登录功能，最轻量级  
**大小**：~3KB  
**特点**：

- 自动网络检测
- 自动登录和重试
- 日志记录
- 错误处理

**适合**：对脚本功能要求不高，追求轻量化的用户

**使用示例**：

```bash
# 编辑脚本，修改USERNAME/PASSWORD等参数
nano campus_network_login.sh

# 执行登录
bash campus_network_login.sh

# 加入crontab
crontab -e  # 添加：*/5 * * * * /path/to/campus_network_login.sh
```

---

#### 2️⃣ `campus_network_login_advanced.sh` - 高级功能脚本

**用途**：完整功能集合，支持配置文件和多种操作模式  
**大小**：~10KB  
**特点**：

- 配置文件支持
- 多种操作模式（--login, --check, --monitor, --status）
- 实时网络监控
- 邮件告警系统
- 彩色输出
- 网络详细信息显示

**适合**：需要高级功能，想要更灵活配置的用户

**使用示例**：

```bash
# 生成配置文件
./campus_network_login_advanced.sh --generate-config

# 执行一次登录
./campus_network_login_advanced.sh --login

# 实时监控网络
./campus_network_login_advanced.sh --monitor

# 查看网络状态
./campus_network_login_advanced.sh --status

# 获取帮助
./campus_network_login_advanced.sh --help
```

---

#### 3️⃣ `install_campus_login.sh` - 自动部署脚本

**用途**：一键自动化部署，完全自动化配置  
**大小**：~8KB  
**特点**：

- 自动依赖检查和安装
- 交互式配置向导
- 自动登录测试
- 自动Crontab配置
- 自动Systemd配置
- 自动日志轮转设置
- 彩色进度反馈

**适合**：第一次使用，想要一键部署的用户

**使用示例**：

```bash
# 需要root权限
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh

# 按照交互式提示输入参数
# 脚本会自动完成所有配置
```

---

### 📚 文档文件（4个）

#### 1️⃣ `README_CAMPUS_LOGIN.md` - 项目总览

**位置**：项目根目录  
**内容**：

- 项目介绍和特性
- 快速开始（5分钟）
- 常用命令
- 配置说明
- 常见问题
- 文档导航

**何时阅读**：第一次接触项目，了解整体情况

---

#### 2️⃣ `QUICK_START.md` - 快速开始指南

**位置**：项目根目录  
**内容**：

- 5分钟快速部署
- 自动部署流程
- 手动配置步骤
- 常见使用场景
- 故障排查要点
- 日志管理基础

**何时阅读**：准备部署脚本时

**导航**：

```
QUICK_START.md
├─ 5分钟快速部署
├─ 获取认证参数
├─ 自动部署步骤
├─ 手动配置步骤
├─ 常见使用场景
├─ 日志管理
└─ 常见问题
```

---

#### 3️⃣ `CAMPUS_LOGIN_GUIDE.md` - 详细使用指南

**位置**：项目根目录  
**内容**：

- 完整的配置教程
- 详细的参数说明
- Crontab和Systemd配置详解
- 日志管理和轮转
- 完整故障排查指南
- 高级配置方案
- 安全建议

**何时阅读**：需要详细信息，遇到问题需要排查

**导航**：

```
CAMPUS_LOGIN_GUIDE.md
├─ 前置准备
├─ 脚本配置步骤
├─ 脚本使用
├─ 开机自启（crontab和systemd）
├─ 日志管理
├─ 故障排查
├─ 高级配置
└─ 常见问题
```

---

#### 4️⃣ `SCRIPTS_COMPARISON.md` - 脚本功能对比

**位置**：项目根目录  
**内容**：

- 三个脚本的详细功能对比表
- 选择建议和推荐
- 快速开始（按脚本分类）
- 脚本之间的关系图
- 配置文件位置汇总
- 安全性说明
- 性能对比
- 学习路径建议
- 故障排查导航

**何时阅读**：不确定用哪个脚本时

**导航**：

```
SCRIPTS_COMPARISON.md
├─ 功能对比表
├─ 选择建议（3个场景）
├─ 快速开始（3种用户）
├─ 脚本功能详解
├─ 脚本间的关系
├─ 配置文件位置
├─ 安全性考虑
└─ 学习路径
```

---

## 🎯 使用场景导航

### 场景1：我是新手，想快速部署

```
推荐文档：README_CAMPUS_LOGIN.md → QUICK_START.md
推荐脚本：install_campus_login.sh

命令：
sudo chmod +x install_campus_login.sh
sudo ./install_campus_login.sh
```

### 场景2：我知道Linux，想自己配置

```
推荐文档：SCRIPTS_COMPARISON.md → QUICK_START.md（手动部分）
推荐脚本：campus_network_login_advanced.sh

命令：
./campus_network_login_advanced.sh --generate-config
nano /etc/campus_login.conf
./campus_network_login_advanced.sh --login
```

### 场景3：我只要最简单的方案

```
推荐文档：README_CAMPUS_LOGIN.md
推荐脚本：campus_network_login.sh

命令：
nano campus_network_login.sh  # 修改参数
./campus_network_login.sh
```

### 场景4：我需要监控和告警

```
推荐文档：CAMPUS_LOGIN_GUIDE.md
推荐脚本：campus_network_login_advanced.sh

命令：
./campus_network_login_advanced.sh --monitor
./campus_network_login_advanced.sh --status
```

### 场景5：我遇到问题，需要排查

```
推荐文档：CAMPUS_LOGIN_GUIDE.md → 故障排查 section

命令：
tail -50 /var/log/campus_login.log
/usr/local/bin/campus_network_login_advanced.sh --status
```

## 📖 文档阅读流程

### 流程1：第一次使用（推荐）

```
1. 阅读 README_CAMPUS_LOGIN.md（5分钟）
   ↓ 了解项目概况

2. 阅读 QUICK_START.md（5分钟）
   ↓ 获取认证参数

3. 选择脚本类型
   - 新手 → 运行 install_campus_login.sh
   - 进阶 → 使用 campus_network_login_advanced.sh

4. 按照脚本提示完成配置
   ↓

5. 查看日志验证运行
   tail -f /var/log/campus_login.log
```

### 流程2：深入学习

```
1. 阅读 SCRIPTS_COMPARISON.md
   ↓ 了解三个脚本的区别

2. 根据选择阅读相应脚本源码
   ↓

3. 根据需要阅读 CAMPUS_LOGIN_GUIDE.md 的高级配置部分
   ↓

4. 尝试自定义和修改脚本
```

### 流程3：遇到问题

```
1. 查看 /var/log/campus_login.log 日志
   ↓

2. 如果日志不清楚，运行诊断
   /usr/local/bin/campus_network_login_advanced.sh --status
   ↓

3. 根据错误类型在 CAMPUS_LOGIN_GUIDE.md 中查找
   ↓

4. 按照故障排查步骤操作
```

## 🚀 快速参考

### 最常用的命令

```bash
# 一键部署
sudo ./install_campus_login.sh

# 手动执行登录
/usr/local/bin/campus_network_login.sh

# 查看日志
tail -f /var/log/campus_login.log

# 检查网络状态
/usr/local/bin/campus_network_login_advanced.sh --check

# 实时监控
/usr/local/bin/campus_network_login_advanced.sh --monitor

# 查看详细状态
/usr/local/bin/campus_network_login_advanced.sh --status
```

### 配置文件位置

```
/etc/campus_login.conf      # 配置文件
/var/log/campus_login.log   # 日志文件
/usr/local/bin/campus_network_login.sh         # 基础脚本
/usr/local/bin/campus_network_login_advanced.sh # 高级脚本
```

### Crontab模板

```bash
# 每分钟检查
* * * * * /usr/local/bin/campus_network_login.sh

# 每5分钟检查（推荐）
*/5 * * * * /usr/local/bin/campus_network_login.sh

# 每10分钟检查
*/10 * * * * /usr/local/bin/campus_network_login.sh

# 开机后60秒运行
@reboot sleep 60 && /usr/local/bin/campus_network_login.sh
```

## 📊 文档速查表

| 需求        | 推荐文档                                 | 关键Section        |
| ----------- | ---------------------------------------- | ------------------ |
| 快速开始    | README_CAMPUS_LOGIN.md                   | 快速开始           |
| 获取参数    | QUICK_START.md                           | 获取认证参数       |
| 自动部署    | QUICK_START.md                           | 自动部署流程       |
| 手动安装    | QUICK_START.md                           | 脚本使用           |
| 脚本选择    | SCRIPTS_COMPARISON.md                    | 选择建议           |
| 详细配置    | CAMPUS_LOGIN_GUIDE.md                    | 脚本配置步骤       |
| Crontab配置 | CAMPUS_LOGIN_GUIDE.md                    | 开机自启配置       |
| Systemd配置 | CAMPUS_LOGIN_GUIDE.md                    | 方案2：使用systemd |
| 日志管理    | CAMPUS_LOGIN_GUIDE.md                    | 日志管理           |
| 故障排查    | CAMPUS_LOGIN_GUIDE.md                    | 故障排查           |
| 高级配置    | CAMPUS_LOGIN_GUIDE.md                    | 高级配置           |
| 常见问题    | README_CAMPUS_LOGIN.md 或 QUICK_START.md | 常见问题           |

## ✨ 项目亮点

### 完整性

✅ 包含基础、高级两个脚本版本  
✅ 配套自动部署脚本  
✅ 四份详细文档  
✅ 完整的错误处理和日志管理

### 易用性

✅ 一键自动部署  
✅ 交互式配置向导  
✅ 详细的使用文档  
✅ 彩色界面和实时反馈

### 灵活性

✅ 支持Crontab定时执行  
✅ 支持Systemd定时器  
✅ 支持配置文件管理  
✅ 支持邮件告警  
✅ 支持实时监控

### 可靠性

✅ 自动重试机制  
✅ 详细日志记录  
✅ 自动日志轮转  
✅ 完整的错误处理  
✅ 网络断线自动恢复

## 📞 获取帮助

### 问题排查步骤

1. 查看最近的日志：`tail -20 /var/log/campus_login.log`
2. 运行诊断：`/usr/local/bin/campus_network_login_advanced.sh --status`
3. 查看相应的文档Section
4. 按照故障排查步骤操作

### 文档查询

- 不知道从哪开始 → README_CAMPUS_LOGIN.md
- 快速部署 → QUICK_START.md
- 选择脚本 → SCRIPTS_COMPARISON.md
- 详细问题 → CAMPUS_LOGIN_GUIDE.md

## 📈 性能指标

| 项目         | 值                |
| ------------ | ----------------- |
| 总包大小     | ~30KB             |
| 脚本运行耗时 | 1-3秒             |
| 内存占用     | 3-10MB            |
| 日志增长速率 | ~1MB/月（未轮转） |
| CPU占用      | 极低（<1%）       |

## 🎓 学习资源

本项目包含：

- 3个完整的Shell脚本示例
- 4份结构化文档
- Crontab和Systemd配置示例
- 日志管理和轮转示例
- 错误处理最佳实践

非常适合学习：

- Shell脚本编写
- Linux系统管理
- 自动化脚本开发
- 系统监控和告警

## 🏆 最佳实践

### 部署

- 使用部署脚本 `install_campus_login.sh` 一键安装
- 验证安装后再配置自启

### 配置

- 选择合适的检查频率（5-10分钟为佳）
- 使用配置文件管理参数
- 定期修改密码时更新配置

### 运维

- 定期检查日志，及时发现问题
- 启用邮件告警，及时收到失败通知
- 使用logrotate自动管理日志

### 安全

- 确保配置文件权限为600
- 定期修改校园网密码
- 不要分享配置文件

## 📝 版本历史

### v1.1 (2024-03-24) - 完整版本

- ✅ 基础脚本 + 高级脚本 + 部署脚本
- ✅ 完整的文档体系
- ✅ 邮件告警支持
- ✅ 实时监控模式
- ✅ 自动部署系统

### v1.0 (2024-03-24) - 初始版本

- ✅ 基础登录脚本
- ✅ 日志管理
- ✅ Crontab支持

---

## 🎉 开始使用

### 推荐步骤：

1. 阅读 [README_CAMPUS_LOGIN.md](README_CAMPUS_LOGIN.md)（3分钟）
2. 阅读 [QUICK_START.md](QUICK_START.md)（5分钟）
3. 运行 `sudo ./install_campus_login.sh`（2分钟）
4. 验证日志 `tail -f /var/log/campus_login.log`

**总耗时：10分钟左右** ⏱️

---

祝你使用愉快！有问题？查看文档或检查日志。🌟
