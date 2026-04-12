# 🖥️ 旧电脑服务器部署指南

> **适用配置**: 8GB 内存 + 普通 CPU（中配）  
> **创建日期**: 2026-03-31  
> **部署方式**: 主要使用 Docker，一键部署

---

## 📋 目录

1. [部署前准备](#1-部署前准备)
2. [必装基础设施](#2-必装基础设施)
3. [个人效率工具](#3-个人效率工具)
4. [影音娱乐系统](#4-影音娱乐系统)
5. [AI 与自动化](#5-ai-与自动化)
6. [开发者工具](#6-开发者工具)
7. [文件与同步](#7-文件与同步)
8. [安全与隐私](#8-安全与隐私)
9. [部署顺序建议](#9-部署顺序建议)
10. [资源占用参考](#10-资源占用参考)

---

## 1. 部署前准备

### 1.1 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 Docker Compose
sudo apt install docker-compose-plugin
```

### 1.2 推荐系统

| 系统 | 推荐度 | 说明 |
|------|--------|------|
| **Ubuntu Server 24.04** | ⭐⭐⭐⭐⭐ | 稳定，教程多 |
| **Debian 12** | ⭐⭐⭐⭐⭐ | 轻量，稳定 |
| **Proxmox VE** | ⭐⭐⭐⭐ | 虚拟化平台，可跑多个系统 |

### 1.3 网络准备

- 固定内网 IP
- 路由器端口转发（如需外网访问）
- 推荐使用 Tailscale/ZeroTier 内网穿透

---

## 2. 必装基础设施

### 2.1 🐳 Portainer - Docker 管理面板

> **GitHub**: [portainer/portainer](https://github.com/portainer/portainer) ⭐ 37k

可视化管理所有 Docker 容器，小白必备！

```bash
docker run -d \
  --name portainer \
  --restart=always \
  -p 9000:9000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce
```

**访问**: `http://服务器IP:9000`

---

### 2.2 📦 Dockge - Docker Compose 管理器

> **GitHub**: [louislam/dockge](https://github.com/louislam/dockge) ⭐ 22.6k

比 Portainer 更专注于 docker-compose 管理，界面更美观。

```bash
mkdir -p /opt/stacks /opt/dockge
cd /opt/dockge

curl -o compose.yaml https://raw.githubusercontent.com/louislam/dockge/master/compose.yaml
docker compose up -d
```

**访问**: `http://服务器IP:5001`

---

### 2.3 🏠 Homepage - 服务导航面板

> **GitHub**: [gethomepage/homepage](https://github.com/gethomepage/homepage) ⭐ 29.2k

漂亮的导航页，一目了然看到所有服务状态。

```yaml
# docker-compose.yml
version: "3.3"
services:
  homepage:
    image: ghcr.io/gethomepage/homepage:latest
    container_name: homepage
    ports:
      - 3000:3000
    volumes:
      - ./config:/app/config
      - /var/run/docker.sock:/var/run/docker.sock:ro
    restart: unless-stopped
```

**备选方案**:
- [Dashy](https://github.com/Lissy93/dashy) ⭐ 24.4k - 功能更丰富
- [Glance](https://github.com/glanceapp/glance) ⭐ 32.9k - RSS 聚合 + 导航

---

### 2.4 📊 Uptime Kuma - 服务监控

> **GitHub**: [louislam/uptime-kuma](https://github.com/louislam/uptime-kuma) ⭐ 84.7k

监控所有服务是否正常运行，支持多种通知方式。

```bash
docker run -d \
  --name uptime-kuma \
  --restart=always \
  -p 3001:3001 \
  -v uptime-kuma:/app/data \
  louislam/uptime-kuma:1
```

**访问**: `http://服务器IP:3001`

---

### 2.5 📈 Beszel - 轻量服务器监控

> **GitHub**: [henrygd/beszel](https://github.com/henrygd/beszel) ⭐ 20.4k

查看 CPU、内存、磁盘等历史数据，带 Docker 统计。

```yaml
version: "3.3"
services:
  beszel:
    image: henrygd/beszel:latest
    container_name: beszel
    ports:
      - 8090:8090
    volumes:
      - ./data:/beszel_data
    restart: unless-stopped
```

---

## 3. 个人效率工具

### 3.1 ☁️ Nextcloud - 私有云盘

> **GitHub**: [nextcloud/server](https://github.com/nextcloud/server) ⭐ 34.5k

替代百度网盘！文件同步、日历、联系人、在线办公。

```yaml
version: "3"
services:
  nextcloud:
    image: nextcloud:latest
    container_name: nextcloud
    ports:
      - 8080:80
    volumes:
      - ./nextcloud:/var/www/html
      - ./data:/var/www/html/data
    environment:
      - MYSQL_HOST=db
      - MYSQL_DATABASE=nextcloud
      - MYSQL_USER=nextcloud
      - MYSQL_PASSWORD=your_password
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: mariadb:10
    container_name: nextcloud-db
    volumes:
      - ./db:/var/lib/mysql
    environment:
      - MYSQL_ROOT_PASSWORD=root_password
      - MYSQL_DATABASE=nextcloud
      - MYSQL_USER=nextcloud
      - MYSQL_PASSWORD=your_password
    restart: unless-stopped
```

**内存占用**: ~500MB

---

### 3.2 🔐 Vaultwarden - 密码管理器

> **GitHub**: [dani-garcia/vaultwarden](https://github.com/dani-garcia/vaultwarden) ⭐ 57.6k

Bitwarden 的轻量替代，管理所有密码。

```bash
docker run -d \
  --name vaultwarden \
  --restart=always \
  -p 8081:80 \
  -v ./vw-data:/data \
  vaultwarden/server:latest
```

**内存占用**: ~50MB（超轻量！）

---

### 3.3 📝 Memos - 轻量笔记

> **GitHub**: [usememos/memos](https://github.com/usememos/memos) ⭐ 30k+

类似 flomo 的碎片化笔记工具。

```bash
docker run -d \
  --name memos \
  --restart=always \
  -p 5230:5230 \
  -v ./memos:/var/opt/memos \
  neosmemo/memos:stable
```

---

## 4. 影音娱乐系统

### 4.1 🎬 Jellyfin - 私人影视库

> **GitHub**: [jellyfin/jellyfin](https://github.com/jellyfin/jellyfin) ⭐ 49.8k

免费开源的 Plex 替代品，打造私人 Netflix！

```yaml
version: "3"
services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    ports:
      - 8096:8096
    volumes:
      - ./config:/config
      - ./cache:/cache
      - /path/to/movies:/media/movies
      - /path/to/tv:/media/tv
    restart: unless-stopped
```

**内存占用**: ~300-500MB（取决于转码）

**配套工具**:
- [Seerr](https://github.com/seerr-team/seerr) ⭐ 10.6k - 影视请求管理
- [Movie_Data_Capture](https://github.com/mvdctop/Movie_Data_Capture) ⭐ 7.4k - 自动刮削

---

### 4.2 🎵 Navidrome - 音乐服务器

> **GitHub**: [navidrome/navidrome](https://github.com/navidrome/navidrome) ⭐ 15k+

私人 Spotify，支持多种客户端。

```yaml
version: "3"
services:
  navidrome:
    image: deluan/navidrome:latest
    container_name: navidrome
    ports:
      - 4533:4533
    volumes:
      - ./data:/data
      - /path/to/music:/music:ro
    environment:
      ND_SCANSCHEDULE: 1h
    restart: unless-stopped
```

**内存占用**: ~100MB

---

### 4.3 📺 TubeArchivist - YouTube 下载归档

> **GitHub**: [tubearchivist/tubearchivist](https://github.com/tubearchivist/tubearchivist) ⭐ 7.7k

自动下载订阅的 YouTube 频道，本地观看。

---

## 5. AI 与自动化

### 5.1 🤖 Open WebUI - 本地 AI 聊天界面

> **GitHub**: [open-webui/open-webui](https://github.com/open-webui/open-webui) ⭐ 129k

最美观的本地 LLM 聊天界面，支持 Ollama。

```bash
# 先安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2:7b

# 再安装 Open WebUI
docker run -d \
  --name open-webui \
  --restart=always \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

**内存占用**: WebUI ~200MB，模型 4-8GB

---

### 5.2 🔗 Langchain-Chatchat - 本地知识库

> **GitHub**: [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) ⭐ 37.7k

基于本地 LLM 的知识库问答系统（中文友好）。

---

### 5.3 ⚡ n8n - 自动化工作流

> **GitHub**: [n8n-io/n8n](https://github.com/n8n-io/n8n) ⭐ 181.9k

开源的 Zapier/IFTTT 替代品，400+ 集成。

```yaml
version: "3"
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    ports:
      - 5678:5678
    volumes:
      - ./n8n_data:/home/node/.n8n
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=your_password
    restart: unless-stopped
```

**用途示例**:
- 每日推送 GitHub Trending 到邮箱
- 监控网站变化并通知
- 自动备份数据

**模板库**: [awesome-n8n-templates](https://github.com/enescingoz/awesome-n8n-templates) ⭐ 20.7k

---

## 6. 开发者工具

### 6.1 🦊 Gitea - 私有 Git 服务

> **GitHub**: [go-gitea/gitea](https://github.com/go-gitea/gitea) ⭐ 54.6k

轻量级 GitHub 替代品，支持 CI/CD。

```bash
docker run -d \
  --name gitea \
  --restart=always \
  -p 3000:3000 \
  -p 222:22 \
  -v ./gitea:/data \
  gitea/gitea:latest
```

**内存占用**: ~150MB

---

### 6.2 🔧 Code-server - 云端 VS Code

> **GitHub**: [coder/code-server](https://github.com/coder/code-server) ⭐ 75k+

在浏览器中使用 VS Code，随时随地写代码。

```bash
docker run -d \
  --name code-server \
  --restart=always \
  -p 8443:8443 \
  -v ./config:/home/coder/.config \
  -v ./project:/home/coder/project \
  -e PASSWORD=your_password \
  codercom/code-server:latest
```

---

## 7. 文件与同步

### 7.1 📁 FileBrowser - 文件管理器

> **GitHub**: [filebrowser/filebrowser](https://github.com/filebrowser/filebrowser) ⭐ 30k+

网页版文件管理器，支持上传下载分享。

```bash
docker run -d \
  --name filebrowser \
  --restart=always \
  -p 8082:80 \
  -v /path/to/files:/srv \
  -v ./filebrowser.db:/database.db \
  filebrowser/filebrowser
```

---

### 7.2 🔄 Syncthing - 文件同步

> **GitHub**: [syncthing/syncthing](https://github.com/syncthing/syncthing) ⭐ 70k+

P2P 文件同步，替代坚果云/Dropbox。

```yaml
version: "3"
services:
  syncthing:
    image: syncthing/syncthing:latest
    container_name: syncthing
    ports:
      - 8384:8384
      - 22000:22000/tcp
      - 22000:22000/udp
    volumes:
      - ./syncthing:/var/syncthing
    restart: unless-stopped
```

---

## 8. 安全与隐私

### 8.1 📷 Immich - 照片备份

> **GitHub**: [immich-app/immich](https://github.com/immich-app/immich) ⭐ 96k

Google Photos 替代品，自动备份手机照片。

```bash
# 使用官方安装脚本
mkdir ./immich-app && cd ./immich-app
curl -o docker-compose.yml https://github.com/immich-app/immich/releases/latest/download/docker-compose.yml
curl -o .env https://github.com/immich-app/immich/releases/latest/download/example.env
docker compose up -d
```

**内存占用**: ~1-2GB（含机器学习）

---

### 8.2 🌐 AdGuard Home - 广告拦截 DNS

> **GitHub**: [AdguardTeam/AdGuardHome](https://github.com/AdguardTeam/AdGuardHome) ⭐ 30k+

全家去广告，替代 Pi-hole。

```bash
docker run -d \
  --name adguardhome \
  --restart=always \
  -p 53:53/tcp -p 53:53/udp \
  -p 3000:3000 \
  -v ./work:/opt/adguardhome/work \
  -v ./conf:/opt/adguardhome/conf \
  adguard/adguardhome
```

---

## 9. 部署顺序建议

### 第一批：基础设施（Day 1）

```
1. Portainer / Dockge    ← Docker 管理
2. Homepage              ← 导航面板
3. Uptime Kuma          ← 服务监控
```

### 第二批：核心服务（Day 2-3）

```
4. Vaultwarden          ← 密码管理（很重要！）
5. Nextcloud            ← 私有云盘
6. AdGuard Home         ← 去广告
```

### 第三批：娱乐系统（Week 1）

```
7. Jellyfin             ← 影视库
8. Navidrome            ← 音乐库
9. Immich               ← 照片备份
```

### 第四批：进阶功能（Week 2+）

```
10. n8n                 ← 自动化
11. Gitea               ← Git 服务
12. Open WebUI + Ollama ← 本地 AI
```

---

## 10. 资源占用参考

| 服务 | 内存占用 | 磁盘占用 | CPU 使用 |
|------|----------|----------|----------|
| Portainer | ~50MB | ~100MB | 低 |
| Homepage | ~100MB | ~50MB | 低 |
| Uptime Kuma | ~150MB | ~100MB | 低 |
| Vaultwarden | ~50MB | ~50MB | 极低 |
| Nextcloud | ~500MB | 按需 | 中 |
| Jellyfin | ~300-500MB | 按需 | 转码时高 |
| n8n | ~300MB | ~500MB | 中 |
| Open WebUI | ~200MB | ~1GB | 低 |
| Ollama (7B模型) | ~4-6GB | ~4GB | 推理时高 |
| Immich | ~1-2GB | 按需 | 中 |

### 8GB 内存分配建议

```
系统预留:     1GB
基础设施:     0.5GB (Portainer + Homepage + Uptime Kuma)
核心服务:     1GB (Vaultwarden + Nextcloud)
影音娱乐:     1GB (Jellyfin + Navidrome)
AI (可选):    4GB (Ollama 7B 模型)
剩余空间:     0.5GB
```

---

## 📚 更多资源

### 必看列表

| 资源 | 链接 | 说明 |
|------|------|------|
| **awesome-selfhosted** | [GitHub](https://github.com/awesome-selfhosted/awesome-selfhosted) ⭐ 283k | 自托管项目大全 |
| **Self-Hosting-Guide** | [GitHub](https://github.com/mikeroyal/Self-Hosting-Guide) ⭐ 19k | 完整部署指南 |
| **ProxmoxVE 脚本** | [GitHub](https://github.com/community-scripts/ProxmoxVE) ⭐ 27.4k | 一键安装脚本 |

### 中文社区

- [r/homelab](https://www.reddit.com/r/homelab/) - Reddit 社区
- [Homelab China](https://t.me/homelab_china) - Telegram 群组
- [V2EX Homelab 节点](https://www.v2ex.com/go/homelab)

---

> 💡 **提示**: 从最感兴趣的项目开始，逐步扩展。不必一次部署所有服务！
> 
> 🔧 **遇到问题**: 搜索 "项目名 + docker" 通常能找到详细教程。
