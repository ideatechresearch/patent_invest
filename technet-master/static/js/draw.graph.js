function updateSize() {
    // 获取窗口的宽度和高度
    width = window.innerWidth;
    height = window.innerHeight;
    // 更新 SVG 元素的宽度和高度
    svg.attr("width", width).attr("height", height);
}


// 定义拖拽行为
function dragstarted(event, d) {
    if (!event.active) {
        simulation.alphaTarget(0.3).restart();//在节点拖拽时设置 alphaTarget 来保持仿真活跃
    }
    d.fx = d.x;
    d.fy = d.y;
}
function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}
function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}


function drawGraph(nodes, edges) {

    // 清除之前的图形
    g.selectAll("*").remove();

    // 初始化力导向图
    //新建一个力导向图，forceManyBody使所有元素相互吸引或排斥。可以设置吸引或排斥的强度
    simulation = d3.forceSimulation()
        //连接力，力的强弱和两节点间的距离成正比.distance(100)每一边的长度
        .force("link", d3.forceLink().id(d => d.id).distance(d => d.value * 200))
        //电荷力，节点之间的相互作用力，如果是正值，则相互吸引，如果是负值，则相互排斥，力的强弱也和节点间的距离有关.strength(-30)
        .force("charge", d3.forceManyBody())
        // 碰撞力，确保节点之间的距离，防止节点重合
        .force("collide", d3.forceCollide(30))
        .force("center", d3.forceCenter(width / 2, height / 2));

    // .force('collision', d3.forceCollide().radius(d => 4))
    //向心力，中心作用力
    //设置图形的中心位置，将元素作为一个整体围绕centering居中

    //生成节点数据
    simulation.nodes(nodes).on("tick", ticked);// 在力导向图更新时调用 ticked 函数
    simulation.force("link").links(edges);  //生成边数据


    //有了节点和边的数据后，绘制边连接线
    var links = g.append("g")
        .selectAll(".link") // 选择所有已有的具有class="link"的元素
        .data(edges, d => `${d.source.id}-${d.target.id}`)
        .enter().append("line")
        .attr("stroke", d => d.rank === 0 ? "#377ba8" : "#999")
        //(d, i) => colorscale(i)
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.7)
        .attr('class', 'link');


    //关系值
    var labels = g.append("g")
        .selectAll(".label")
        .data(edges)
        .enter().append("text")
        .text(d => d.relation)
        .attr('font-size', '10px')  // 设置字体大小
        .attr('font-weight', '100')
        .attr('class', 'label')
        .style('user-select', 'none');// 禁止文本被选中

    //绘制节点，先为节点和节点上的文字分组
    var gs = g.selectAll(".node")
        .data(nodes, d => d.id)
        .enter().append("g")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .attr('class', 'node')
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        )
        .on("click", ClickNode);


    //绘制节点
    gs.append("circle")
        .attr("r", d => d.radius)//20
        .attr("fill", d => colorscale(d.depth))// .attr("fill", "steelblue") return "rgba(255, 0, 0, 0.5)";红色，透明度为 0.5
        .attr("fill-opacity", 0.5);  // 设置半透明效果
    // .classed("node", true)

    //文字
    gs.append("text")
        .attr("dy", ".35em")  // 使文字在节点中心垂直居中
        .text(d => d.name)
        .attr("text-anchor", "middle")
        .attr("fill", "black")
        .attr('pointer-events', 'none');// 防止文本元素影响节点的拖拽事件


    // 更新节点和连接的位置
    function ticked() {
        links
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        labels
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

        gs
            .attr("transform", d => `translate(${d.x},${d.y})`);
    }
}

// Function to update the graph with new nodes and links
function updateSimulation(nodes, edges) {
    simulation.nodes(nodes).on("tick", ticked);
    simulation.force("link").links(edges);

    var links = g.selectAll(".link")
        .data(edges, d => `${d.source.id}-${d.target.id}`);

    links.exit().remove();

    links = links.enter().append("line")
        .attr("stroke", d => d.rank === 0 ? "#377ba8" : "#999")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.7)
        .attr('class', 'link')
        .merge(links);

    var labels = g.selectAll(".label")//.data(edges)
        .data(edges, d => `${d.source.id}-${d.target.id}`);

    labels.exit().remove();

    labels = labels.enter().append("text")
        .text(d => d.relation)
        .attr('font-size', '10px')  // 设置字体大小
        .attr('font-weight', '100')
        .attr('class', 'label')
        .style('user-select', 'none')// 禁止文本被选中
        .merge(labels);

    var gs = g.selectAll(".node").data(nodes, d => d.id);

    gs.exit().remove();

    var gsEnter = gs.enter().append("g")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .attr('class', 'node')
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        )
        .on("click", ClickNode);
    // .merge(node);


    gsEnter.append("circle")
        .attr("r", d => d.radius)
        .attr("fill", d => colorscale(d.depth))
        .attr("fill-opacity", 0.5);

    gsEnter.append("text")
        .attr("dy", ".35em")
        .text(d => d.name)
        .attr("text-anchor", "middle")
        .attr("fill", "black")
        .attr('pointer-events', 'none');

    gs = gsEnter.merge(gs);

    // 更新节点和连接的位置
    function ticked() {
        links
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        labels
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

        gs
            .attr("transform", d => `translate(${d.x},${d.y})`);
    }
    // 重新启动仿真
    simulation.alpha(0.7).restart();
}

function fetchDataAndDraw(url) {
    // 如果正在进行数据请求，则不执行新的请求
    if (fetchingData) {
        console.log("数据请求正在进行中...");
        return;
    }

    // 将标志变量设置为true，表示开始进行数据请求
    fetchingData = true;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            //\[{}, {}, {}, {}, {}\]
            nodes = data.nodes;
            edges = data.edges;

            // Initialize node positions
            nodes.forEach(node => {
                if (node.depth === undefined) { node.depth = 0; }
                if (node.radius === undefined) { node.radius = 20; }

                // if (node.x === undefined || node.y === undefined) {
                //     node.x = Math.random() * width;
                //     node.y = Math.random() * height;
                // }
            });
            // Ensure edges have source and target as node objects
            edges.forEach(edge => {
                if (edge.rank === undefined) { edge.rank = 0; }
                edge.source = nodes.find(node => node.id === edge.source.id || node.id === edge.source) || edge.source;
                edge.target = nodes.find(node => node.id === edge.target.id || node.id === edge.target) || edge.target;
            });
            //在浏览器的控制台输出
            console.log(nodes);
            console.log(edges);
            drawGraph(nodes, edges);
        })
        .catch(error => {
            console.error('数据请求错误:', error);
        })
        .finally(() => {
            // 无论请求成功或失败，都将标志变量设置为false，以便允许新的请求
            fetchingData = false;
        });
}

// Function to fetch new node relations from API
function fetchNodeRelations(url) {
    // 将所有节点的完整信息包含在请求体中
    var requestBody = {
        existingNodes: nodes.map(n => ({
            id: n.id,
            name: n.name,
            depth: n.depth
        }))
    };

    // console.log("Request Body:", requestBody);

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    })
        .then(response => response.json())
        .then(data => {

            var newnodes = data.nodes;
            var newedges = data.edges;
            //在浏览器的控制台输出
            console.log(newnodes);
            console.log(newedges);
            // Add new nodes to the existing nodes
            newnodes.forEach(node => {
                if (node.x === undefined || node.y === undefined) {
                    node.x = Math.random() * width;
                    node.y = Math.random() * height;
                }
                if (node.depth === undefined) { node.depth = 0; }
                if (node.radius === undefined) { node.radius = 20; }
                if (!nodes.find(n => n.id === node.id)) {
                    nodes.push(node);
                }
            });

            // Add new links to the existing links
            newedges.forEach(edge => {
                if (edge.rank === undefined) { edge.rank = 0; }
                edge.source = nodes.find(node => node.id === edge.source.id || node.id === edge.source) || edge.source;
                edge.target = nodes.find(node => node.id === edge.target.id || node.id === edge.target) || edge.target;
                if (!edges.find(l => l.source.id === edge.source && l.target.id === edge.target)) {
                    edges.push(edge);
                }
            });
            // 更新图形updateGraph
            updateSimulation(nodes, edges);
        })
        .catch(error => {
            console.error('数据请求错误:', error);
        });
}

function fetchCytoscape(url) {
    // <div id="cy" style="width: 800px; height: 600px;"></div>
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: 'MATCH (n)-[r]->(m) RETURN n, r, m' })
    })
        .then(response => response.json())
        .then(data => {
            var cy = cytoscape({
                container: document.getElementById('cy'),
                elements: [
                    ...data.nodes.map(node => ({
                        data: { id: node.id, label: node.labels[0], ...node.properties }
                    })),
                    ...data.relationships.map(rel => ({
                        data: { id: rel.id, source: rel.start, target: rel.end, label: rel.type, ...rel.properties }
                    }))
                ],
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'background-color': '#61bffc'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'label': 'data(label)',
                            'width': 3,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle'
                        }
                    }
                ],
                layout: {
                    name: 'grid',
                    rows: 1
                }
            });
        });
}

function parseLayers(layersString) {
    // 正则分割字符串，保留数字，过滤大于0的整数
    const nums = layersString.split(/[^\w\s]| /).filter(i => /^\d+$/.test(i) && parseInt(i) > 0).map(i => parseInt(i));
    return nums;
}

function calcLayersTotal(layers = [3, 3, 3]) {
    let n = 0;
    let product = 1;
    for (let layer of layers) {
        n += product;
        product *= layer;
    }
    return n;
}
function calcNodesTotal(layers = [3, 3, 3]) {
    let n = 1;
    let product = 1;
    for (let layer of layers) {
        product *= layer;
        n += product;
    }
    return n;
}