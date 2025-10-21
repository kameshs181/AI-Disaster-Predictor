async function predict() {
    const city = document.getElementById('city').value;
    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({city})
    });
    const data = await res.json();
    if(data.error){
        alert(data.error);
        return;
    }
    document.getElementById('flood-prob').innerText = data.flood_prob;
    document.getElementById('flood-level').innerText = data.flood_risk;
    document.getElementById('cyclone-prob').innerText = data.cyclone_prob;
    document.getElementById('cyclone-level').innerText = data.cyclone_risk;

    updateChart();
}

async function updateChart(){
    const res = await fetch('/recent_predictions');
    const predictions = await res.json();
    const labels = predictions.map(p=>p.city);
    const floodData = predictions.map(p=>p.flood_prob);
    const cycloneData = predictions.map(p=>p.cyclone_prob);

    const ctx = document.getElementById('predictionChart').getContext('2d');
    new Chart(ctx, {
        type:'bar',
        data:{
            labels: labels,
            datasets:[
                {label:'Flood Probability', data:floodData, backgroundColor:'rgba(54,162,235,0.6)'},
                {label:'Cyclone Probability', data:cycloneData, backgroundColor:'rgba(255,99,132,0.6)'}
            ]
        },
        options:{responsive:true, scales:{y:{beginAtZero:true}}}
    });
}
