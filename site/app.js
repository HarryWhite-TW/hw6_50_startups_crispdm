const stateSummary = [
  { state: "California", rows: 13, rd: 98569, administration: 104205, marketing: 281608, profit: 150409 },
  { state: "Florida", rows: 16, rd: 92224, administration: 84359, marketing: 228082, profit: 136776 },
  { state: "New York", rows: 21, rd: 85357, administration: 105034, marketing: 202585, profit: 135186 }
];

const metrics = [
  ["Test R2 Score", "0.9460"],
  ["Test MAE", "5,578.56"],
  ["Test RMSE", "6,971.82"],
  ["CV Mean R2", "0.9694"],
  ["CV Std", "0.0165"]
];

const featureSelection = {
  labels: [1, 2, 3, 4, 5],
  methods: [
    { name: "SFS / Forward", color: "#2f6b7c", rmse: [11638, 6972, 7058, 6908, 6972], r2: [0.849, 0.946, 0.945, 0.947, 0.946] },
    { name: "RFE", color: "#b95050", rmse: [51635, 51708, 11666, 6972, 6972], r2: [-1.963, -1.971, 0.849, 0.946, 0.946] },
    { name: "SelectKBest", color: "#4c72b0", rmse: [11638, 6972, 7058, 6908, 6972], r2: [0.849, 0.946, 0.945, 0.947, 0.946] },
    { name: "Lasso", color: "#4f8a70", rmse: [11638, 6972, 6991, 7058, 6972], r2: [0.849, 0.946, 0.946, 0.945, 0.946] },
    { name: "Random Forest", color: "#8172b3", rmse: [11638, 6972, 7022, 6991, 6972], r2: [0.849, 0.946, 0.945, 0.946, 0.946] }
  ]
};

const money = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);

const metricCards = document.getElementById("metricCards");
metrics.forEach(([label, value]) => {
  const card = document.createElement("article");
  card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
  metricCards.appendChild(card);
});

const tableBody = document.getElementById("stateTable");
const stateFilter = document.getElementById("stateFilter");

let profitChart;
let spendingChart;

function filteredRows() {
  const selected = stateFilter.value;
  return selected === "All" ? stateSummary : stateSummary.filter((row) => row.state === selected);
}

function renderTable(rows) {
  tableBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.state}</td>
          <td>${row.rows}</td>
          <td>${money(row.rd)}</td>
          <td>${money(row.marketing)}</td>
          <td>${money(row.profit)}</td>
        </tr>
      `
    )
    .join("");
}

function renderExplorer() {
  const rows = filteredRows();
  renderTable(rows);

  profitChart.data.labels = rows.map((row) => row.state);
  profitChart.data.datasets[0].data = rows.map((row) => row.profit);
  profitChart.update();

  const totals = rows.reduce(
    (acc, row) => {
      acc.rows += row.rows;
      acc.rd += row.rd * row.rows;
      acc.administration += row.administration * row.rows;
      acc.marketing += row.marketing * row.rows;
      return acc;
    },
    { rows: 0, rd: 0, administration: 0, marketing: 0 }
  );

  spendingChart.data.datasets[0].data = [
    totals.rd / totals.rows,
    totals.administration / totals.rows,
    totals.marketing / totals.rows
  ];
  spendingChart.update();
}

function lineDatasets(metricName) {
  return featureSelection.methods.map((method) => ({
    label: method.name,
    data: method[metricName],
    borderColor: method.color,
    backgroundColor: method.color,
    pointRadius: 4,
    borderWidth: 2,
    tension: 0.25
  }));
}

function createCharts() {
  profitChart = new Chart(document.getElementById("profitByStateChart"), {
    type: "bar",
    data: {
      labels: [],
      datasets: [{ label: "Average Profit", data: [], backgroundColor: "#2f6b7c" }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { ticks: { callback: (value) => money(value) } } }
    }
  });

  spendingChart = new Chart(document.getElementById("spendingProfileChart"), {
    type: "bar",
    data: {
      labels: ["R&D Spend", "Administration", "Marketing Spend"],
      datasets: [{ label: "Average Spending", data: [], backgroundColor: ["#2f6b7c", "#c9822e", "#4f8a70"] }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { ticks: { callback: (value) => money(value) } } }
    }
  });

  new Chart(document.getElementById("rmseChart"), {
    type: "line",
    data: { labels: featureSelection.labels, datasets: lineDatasets("rmse") },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { title: { display: true, text: "Test RMSE" } } }
    }
  });

  new Chart(document.getElementById("r2Chart"), {
    type: "line",
    data: { labels: featureSelection.labels, datasets: lineDatasets("r2") },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { title: { display: true, text: "Test R-squared" } } }
    }
  });
}

createCharts();
renderExplorer();
stateFilter.addEventListener("change", renderExplorer);
