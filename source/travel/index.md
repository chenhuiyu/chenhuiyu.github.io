---
title: travel
type: "travel"
---
Wanderlust Adventures
---

<!-- 引入 Swiper 的 CSS -->
<link rel="stylesheet" href="https://unpkg.com/swiper/swiper-bundle.min.css" />

<!-- 引入 Leaflet 的 CSS -->
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />

<style>
/* 为地图容器和Swiper设置样式，防止样式冲突 */
#map {
    height: 500px;
    width: 100%;
    position: relative;
}
.custom-popup .swiper-slide img {
    width: 200px;
    height: 200px;  /* 这里设置图片的高度 */
    object-fit: cover;  /* 这里设置图片自动裁剪并填充整个设定的区域 */
}
</style>

<div id="map"></div>

<!-- 引入 Swiper 的 JavaScript 文件 -->
<script src="https://unpkg.com/swiper/swiper-bundle.min.js"></script>

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
                    var radius = getRadius(feature.properties.visits);
                    return L.circleMarker(latlng, { fillColor: color, fillOpacity: 0.5, radius: radius });
                },
                onEachFeature: function (feature, layer) {
                    var popupContent = `
                        <div class="custom-popup">
                            <h2>${feature.properties.name}</h2>
                            <p>${feature.properties.description}</p>
                            <div class="swiper-container">
                                <div class="swiper-wrapper">`;

                    feature.properties.images.forEach(function(image) {
                        popupContent += `<div class="swiper-slide"><img src="${image}" /></div>`;
                    });

                    popupContent += `
                                </div>
                                <div class="swiper-button-next"></div>
                                <div class="swiper-button-prev"></div>
                            </div>
                        </div>`;

                    layer.bindPopup(popupContent);

                    layer.on('popupopen', function() {
                        new Swiper('.swiper-container', {
                            navigation: {
                                nextEl: '.swiper-button-next',
                                prevEl: '.swiper-button-prev',
                            },
                        });
                    });
                }
            }).addTo(map);
        });

    function getColor(visits) {
        return visits > 10 ? '#800026' :
               visits > 5  ? '#BD0026' :
               visits > 2  ? '#E31A1C' :
               visits > 1  ? '#FC4E2A' :
                             '#FFEDA0';
    }

    function getRadius(visits) {
        return visits > 10 ? 10 :
               visits > 5  ? 8 :
               visits > 2  ? 6 :
               visits > 1  ? 4 :
                             2;
    }

    setTimeout(function() {
        map.invalidateSize();
    }, 100);
});
</script>
