import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # CDN内容分发网络原理详解

        ## 让全球用户都能快速访问你的内容

        ![CDN概念图](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&h=400&fit=crop)

        ---

        ### 🌐 什么是CDN？

        **CDN（Content Delivery Network，内容分发网络）** 是一种分布式网络架构，通过在全球各地部署边缘服务器，将内容缓存到离用户最近的节点，从而提高内容访问速度和用户体验。

        **核心理念：** 让内容更接近用户，减少网络延迟

        ### 🎯 CDN解决的核心问题

        **📍 地理距离问题**
        - 用户与源服务器距离过远导致延迟高
        - 跨国访问时的网络路由复杂

        **⚡ 性能瓶颈**
        - 源服务器负载过重
        - 带宽限制影响访问速度

        **🛡️ 可靠性挑战**
        - 单点故障风险
        - 网络拥塞导致的服务中断

        **💰 成本控制**
        - 减少源服务器带宽消耗
        - 降低基础设施投入
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🏗️ CDN基础架构

        CDN网络由多个关键组件构成，下面通过图表来理解其整体架构：
        """
    )
    return


@app.cell
def __(mo):
    # CDN整体架构图
    mo.mermaid(
        """
        graph TB
            User[👤 用户] --> DNS[🌐 DNS解析]
            DNS --> CDN_DNS[📡 CDN DNS服务器]
            CDN_DNS --> Edge1[🏢 边缘节点1<br/>北京]
            CDN_DNS --> Edge2[🏢 边缘节点2<br/>上海]
            CDN_DNS --> Edge3[🏢 边缘节点3<br/>广州]
            
            Edge1 --> Origin[🏛️ 源服务器<br/>Origin Server]
            Edge2 --> Origin
            Edge3 --> Origin
            
            subgraph "CDN网络"
                Edge1
                Edge2
                Edge3
                CDN_DNS
            end
            
            subgraph "内容源"
                Origin
            end
            
            style User fill:#e1f5fe
            style Origin fill:#fff3e0
            style Edge1 fill:#f3e5f5
            style Edge2 fill:#f3e5f5
            style Edge3 fill:#f3e5f5
            style CDN_DNS fill:#e8f5e8
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🔄 CDN工作流程详解

        让我们通过一个具体的例子来理解CDN是如何工作的：
        """
    )
    return


@app.cell
def __(mo):
    # CDN工作流程图
    mo.mermaid(
        """
        sequenceDiagram
            participant U as 👤 用户(北京)
            participant DNS as 🌐 本地DNS
            participant CDNS as 📡 CDN DNS
            participant Edge as 🏢 边缘节点(北京)
            participant Origin as 🏛️ 源服务器(美国)
            
            Note over U,Origin: 用户请求 www.example.com/image.jpg
            
            U->>DNS: 1. 域名解析请求
            DNS->>CDNS: 2. 查询CDN DNS
            CDNS->>CDNS: 3. 智能调度<br/>选择最优节点
            CDNS->>DNS: 4. 返回边缘节点IP
            DNS->>U: 5. 返回IP地址
            
            U->>Edge: 6. 请求内容
            
            alt 缓存命中
                Edge->>U: 7a. 直接返回缓存内容
            else 缓存未命中
                Edge->>Origin: 7b. 回源请求
                Origin->>Edge: 8. 返回原始内容
                Edge->>Edge: 9. 缓存内容
                Edge->>U: 10. 返回内容给用户
            end
            
            Note over U,Origin: 整个过程大大减少了延迟
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🧠 CDN核心技术原理

        #### 1. 智能DNS调度

        **📍 地理位置调度**
        - 根据用户IP地址判断地理位置
        - 选择距离最近的边缘节点
        - 考虑网络拓扑和运营商线路

        **⚡ 性能调度**
        - 实时监控各节点负载状况
        - 选择响应时间最短的节点
        - 动态调整流量分配

        **🔄 健康检查**
        - 定期检测节点可用性
        - 自动剔除故障节点
        - 实现故障转移

        #### 2. 缓存策略

        **🕒 缓存时间控制**
        ```http
        # HTTP缓存头示例
        Cache-Control: max-age=3600, public
        Expires: Wed, 21 Oct 2024 07:28:00 GMT
        ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
        Last-Modified: Wed, 21 Oct 2024 06:28:00 GMT
        ```

        **📊 缓存层级**

        - L1缓存：内存缓存（毫秒级访问）
        - L2缓存：SSD缓存（微秒级访问）
        - L3缓存：机械硬盘（毫秒级访问）

        **🎯 缓存策略**

        - **LRU（最近最少使用）**：淘汰最久未访问的内容
        - **LFU（最少使用频率）**：淘汰访问频率最低的内容
        - **TTL（生存时间）**：基于时间的自动过期
        """
    )
    return


@app.cell
def __(mo):
    # 缓存层级结构图
    mo.mermaid(
        """
        graph TD
            Request[📥 用户请求] --> L1{🧠 L1内存缓存<br/>命中率: 85%}
            L1 -->|命中| Return1[⚡ 1ms响应]
            L1 -->|未命中| L2{💾 L2 SSD缓存<br/>命中率: 12%}
            L2 -->|命中| Return2[⚡ 10ms响应]
            L2 -->|未命中| L3{🗄️ L3机械硬盘<br/>命中率: 2.5%}
            L3 -->|命中| Return3[⚡ 50ms响应]
            L3 -->|未命中| Origin[🏛️ 回源请求<br/>命中率: 0.5%]
            Origin --> Return4[⏰ 200ms+响应]
            
            style L1 fill:#e8f5e8
            style L2 fill:#fff3e0
            style L3 fill:#fce4ec
            style Origin fill:#ffebee
            style Return1 fill:#c8e6c9
            style Return2 fill:#dcedc8
            style Return3 fill:#f8bbd9
            style Return4 fill:#ffcdd2
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🌍 CDN节点分布策略

        CDN的效果很大程度上取决于节点的分布策略：
        """
    )
    return


@app.cell
def __(mo):
    # 全球CDN节点分布图
    mo.mermaid(
        """
        graph TB
            subgraph "🌏 亚太地区"
                Beijing[🏢 北京节点<br/>延迟: 5ms]
                Shanghai[🏢 上海节点<br/>延迟: 8ms]
                Tokyo[🏢 东京节点<br/>延迟: 15ms]
                Singapore[🏢 新加坡节点<br/>延迟: 25ms]
            end
            
            subgraph "🌍 欧洲地区"
                London[🏢 伦敦节点<br/>延迟: 120ms]
                Frankfurt[🏢 法兰克福节点<br/>延迟: 130ms]
            end
            
            subgraph "🌎 美洲地区"
                NewYork[🏢 纽约节点<br/>延迟: 180ms]
                LosAngeles[🏢 洛杉矶节点<br/>延迟: 160ms]
            end
            
            subgraph "🏛️ 源服务器"
                Origin[源服务器<br/>美国西海岸]
            end
            
            User[👤 中国用户] --> Beijing
            User --> Shanghai
            
            Beijing -.-> Origin
            Shanghai -.-> Origin
            Tokyo -.-> Origin
            Singapore -.-> Origin
            London -.-> Origin
            Frankfurt -.-> Origin
            NewYork -.-> Origin
            LosAngeles -.-> Origin
            
            style User fill:#e1f5fe
            style Beijing fill:#c8e6c9
            style Shanghai fill:#c8e6c9
            style Origin fill:#fff3e0
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 📈 CDN性能优化技术

        #### 1. 内容预取和预热

        **🔥 缓存预热**

        - 在内容发布前主动推送到边缘节点
        - 避免首次访问时的回源延迟
        - 适用于热门内容和新发布内容

        **🎯 智能预取**

        - 基于用户行为预测下一步访问内容
        - 利用机器学习算法优化预取策略
        - 在用户空闲时间进行预取

        #### 2. 内容压缩和优化

        **📦 压缩技术**
        ```
        原始文件: 1MB
        ↓ Gzip压缩
        压缩后: 200KB (压缩率80%)
        ↓ Brotli压缩  
        压缩后: 150KB (压缩率85%)
        ```

        **🖼️ 图像优化**

        - WebP格式转换（减少30-50%文件大小）
        - 自适应图像质量调整
        - 响应式图像尺寸适配

        #### 3. 协议优化

        **🚀 HTTP/2 和 HTTP/3**

        - 多路复用减少连接数
        - 服务器推送技术
        - QUIC协议降低延迟

        **🔒 TLS优化**

        - TLS 1.3快速握手
        - 会话复用技术
        - OCSP装订优化
        """
    )
    return


@app.cell
def __(mo):
    # CDN优化技术对比图
    mo.mermaid(
        """
        graph LR
            subgraph "传统方式"
                A1[用户请求] --> A2[DNS解析 100ms]
                A2 --> A3[建立连接 200ms]
                A3 --> A4[下载内容 2000ms]
                A4 --> A5[总耗时: 2300ms]
            end
            
            subgraph "CDN优化后"
                B1[用户请求] --> B2[智能DNS 20ms]
                B2 --> B3[就近连接 50ms]
                B3 --> B4[缓存内容 100ms]
                B4 --> B5[总耗时: 170ms]
            end
            
            subgraph "性能提升"
                C1[⚡ 速度提升: 13.5倍]
                C2[📊 延迟降低: 92.6%]
                C3[💰 带宽节省: 80%]
            end
            
            style A5 fill:#ffcdd2
            style B5 fill:#c8e6c9
            style C1 fill:#e8f5e8
            style C2 fill:#e8f5e8
            style C3 fill:#e8f5e8
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🛡️ CDN安全防护机制

        CDN不仅提升性能，还提供多层安全防护：
        """
    )
    return


@app.cell
def __(mo):
    # CDN安全防护架构图
    mo.mermaid(
        """
        graph TD
            Internet[🌐 互联网流量] --> WAF[🛡️ Web应用防火墙<br/>SQL注入/XSS防护]
            WAF --> DDoS[⚔️ DDoS防护<br/>流量清洗]
            DDoS --> RateLimit[🚦 频率限制<br/>防止滥用]
            RateLimit --> GeoBlock[🌍 地理封锁<br/>IP黑白名单]
            GeoBlock --> SSL[🔒 SSL/TLS加密<br/>数据传输安全]
            SSL --> Edge[🏢 边缘节点<br/>安全内容分发]
            Edge --> Origin[🏛️ 源服务器<br/>隐藏真实IP]
            
            subgraph "安全层级"
                WAF
                DDoS
                RateLimit
                GeoBlock
                SSL
            end
            
            style WAF fill:#ffebee
            style DDoS fill:#fce4ec
            style RateLimit fill:#f3e5f5
            style GeoBlock fill:#ede7f6
            style SSL fill:#e8eaf6
            style Edge fill:#e3f2fd
            style Origin fill:#e0f2f1
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 📊 CDN应用场景与效果

        #### 主要应用场景

        **🎮 游戏行业**

        - 游戏客户端下载加速
        - 游戏更新包分发
        - 实时游戏数据同步

        **📺 视频流媒体**

        - 视频点播加速
        - 直播流分发
        - 自适应码率传输

        **🛒 电商平台**

        - 商品图片加速
        - 静态资源优化
        - 购物高峰期负载均衡

        **📱 移动应用**

        - APP下载加速
        - API接口加速
        - 移动端资源优化

        #### 性能提升效果

        | 指标 | 优化前 | 优化后 | 提升幅度 |
        |------|--------|--------|----------|
        | 🕒 页面加载时间 | 3.2秒 | 0.8秒 | **75%** |
        | 📊 首字节时间(TTFB) | 800ms | 120ms | **85%** |
        | 💾 带宽使用 | 100% | 20% | **80%** |
        | 🎯 缓存命中率 | 0% | 95% | **95%** |
        | 👥 并发用户数 | 1,000 | 10,000 | **10倍** |

        #### 成本效益分析

        **💰 成本节省**

        - 源服务器带宽费用减少80%
        - 服务器硬件投入减少60%
        - 运维人力成本降低40%

        **📈 业务价值**

        - 用户体验提升带来转化率增长15-25%
        - 搜索引擎排名提升（页面速度是SEO因素）
        - 全球业务扩展能力增强
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 🔧 CDN选择与配置指南

        #### 主流CDN服务商对比

        **🌟 国际CDN**

        - **Cloudflare**: 全球节点最多，免费套餐丰富
        - **AWS CloudFront**: 与AWS生态深度集成
        - **Google Cloud CDN**: 机器学习优化，性能出色

        **🇨🇳 国内CDN**

        - **阿里云CDN**: 国内覆盖最全，价格合理
        - **腾讯云CDN**: 社交场景优化，游戏加速强
        - **百度云CDN**: AI技术加持，智能调度

        #### 配置最佳实践

        **⚙️ 缓存配置**
        ```nginx
        # 静态资源缓存配置示例
        location ~* \.(jpg|jpeg|png|gif|css|js)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            add_header Vary "Accept-Encoding";
        }
        
        # HTML文件缓存配置
        location ~* \.html$ {
            expires 1h;
            add_header Cache-Control "public, must-revalidate";
        }
        ```

        **🎯 性能监控**

        - 设置缓存命中率监控（目标>90%）
        - 监控回源率（目标<10%）
        - 跟踪用户访问延迟
        - 定期进行性能测试

        ### 🚀 CDN未来发展趋势

        **🤖 AI驱动优化**

        - 智能缓存策略
        - 预测性内容预取
        - 自动化性能调优

        **🌐 边缘计算集成**

        - 边缘函数计算
        - 实时数据处理
        - IoT设备就近服务

        **🔒 安全能力增强**

        - 零信任网络架构
        - 实时威胁检测
        - 隐私保护技术

        **📱 5G时代适配**

        - 超低延迟优化
        - 移动边缘计算
        - 实时交互应用支持

        ---

        ### 💡 总结

        CDN作为现代互联网基础设施的重要组成部分，通过智能的内容分发和缓存策略，显著提升了用户体验和系统性能。随着技术的不断发展，CDN正在向更智能、更安全、更高效的方向演进，成为数字化时代不可或缺的技术支撑。

        **关键要点：**
        
        - 🎯 **核心价值**: 让内容更接近用户，减少延迟
        - 🏗️ **技术基础**: 分布式架构 + 智能调度 + 多层缓存
        - 🛡️ **安全防护**: 多层安全机制保护源服务器
        - 📈 **业务价值**: 性能提升 + 成本节省 + 用户体验优化
        """
    )
    return


if __name__ == "__main__":
    app.run()
