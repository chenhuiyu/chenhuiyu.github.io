---
title: travel
type: "travel"
---
Wanderlust Adventures
---

<!-- 引入 Leaflet 的 CSS -->
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />

<style>
    /* 为地图容器设置样式，防止样式冲突 */
    #map {
        height: 500px;
        width: 100%;
        position: relative;
    }
</style>

<div id="map"></div>

<!-- 引入 Leaflet 的 JavaScript 文件 -->
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = L.map('map').setView([0, 0], 2);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);

        fetch('/travel/Geo.json')
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
