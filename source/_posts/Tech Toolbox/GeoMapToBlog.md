---
title: 在 Blog 中添加可交互式足迹地图
date: 2023-07-08 13:40:21
categories:
- Tech Toolbox
tags:
- Blog
---
# 在Blog中添加可交互式足迹地图

这篇文章将向你展示如何在基于Hexo和Next的GitHub Pages博客中创建一个交互式的世界地图页面，这个地图将展示你曾经访问过的城市，你可以根据你对每个城市访问的频率在地图上显示不同颜色的标记，你还可以点击这些标记来显示更多关于这个城市的信息。

## 准备工作

你需要安装以下工具：

- Node.js 和 NPM
- Hexo

确保你的博客已经被部署到GitHub Pages，并且你在本地的开发环境已经正确设置。

## 步骤一：创建新页面

在你的Hexo项目的根目录下，运行以下命令：

```bash
hexo new page "travel"
```

这个命令将在`source`目录下创建一个名为"travel"的文件夹，并在该文件夹下创建一个`index.md`文件。

## 步骤二：安装 Leaflet

在你的Hexo项目的根目录下，运行以下命令：

```bash
npm install leaflet
```

然后在`index.md`文件的最顶部引入Leaflet的CSS和JS：

```markdown
---
title: Travel
---

<!-- 引入 Leaflet 的 CSS -->
<link rel="stylesheet" href="/node_modules/leaflet/dist/leaflet.css" />

<!-- 引入 Leaflet 的 JavaScript 文件 -->
<script src="/node_modules/leaflet/dist/leaflet.js"></script>
```

## 步骤三：创建地图容器

在`index.md`中，创建一个用来承载地图的`<div>`元素，然后通过CSS给它设置一个明确的高度：

```markdown
<style>
    #map {
        height: 500px;
        width: 100%;
        position: relative;
    }
</style>

<div id="map"></div>
```

## 步骤四：初始化地图

使用Leaflet的API初始化地图，并将其中心设置为经度0,纬度0（大西洋中部），并设置初始的缩放级别为2。

```markdown
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = L.map('map').setView([0, 0], 2);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);
    });
</script>
```

## 步骤五：添加城市数据

创建一个GeoJSON文件来存储城市的数据，包括城市的名称，经纬度，你访问的次数，以及一个图片的URL。文件的格式应该如下：

```json
{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "北京",
                "visits": 3,
                "image": "https://your-image-url.com"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [116.4074, 39.9042]
            }
        },
        ...
    ]
}
```

将此文件保存在你的Hexo项目的`source`目录下，例如，你可以命名为`cities.json`。

## 步骤六：加载数据并显示在地图上

在`index.md`文件中，添加以下代码来加载GeoJSON文件，然后在地图上显示数据：

```markdown
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = L.map('map').setView([0, 0], 2);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);

        fetch('/travel/cities.json')
            .then(response => response.json())
            .then(data => {
                L.geoJSON(data, {
                    pointToLayer: function (feature, latlng) {
                        var color = getColor(feature.properties.visits);
                        return L.circleMarker(latlng, { fillColor: color, fillOpacity: 0.5 });
                    },
                    onEachFeature: function (feature, layer) {
                        layer.bindPopup(`<h2>${feature.properties.name}</h2><img src="${feature.properties.image}" width="200">`);
                    }
                }).addTo(map);
            });

        function getColor(visits) {
            // 根据你访问的次数返回不同的颜色
            return visits > 10 ? '#800026' :
                   visits > 5  ? '#BD0026' :
                   visits > 2  ? '#E31A1C' :
                   visits > 1  ? '#FC4E2A' :
                                 '#FFEDA0';
        }

        // 强制Leaflet重新计算地图的尺寸
        setTimeout(function() {
            map.invalidateSize();
        }, 100);
    });
</script>
```

以上的JavaScript代码在页面加载完成后会运行，首先初始化地图，然后加载并解析GeoJSON文件，对于每一个城市，创建一个圆形的标记，颜色根据访问次数决定。当点击标记时，弹出一个包含城市名称和图片的窗口。

至此，你的博客就成功地添加了一个交互式的足迹地图页面。
