const pptxgen = require('pptxgenjs');
const html2pptx = require('C:/Users/86139/.config/opencode/skills/pptx/scripts/html2pptx.js');
const path = require('path');

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Zhang & Luo | OR-LLM-Agent';
  pptx.title = '基于LLM的ALNS算法求解VRP问题';
  pptx.subject = 'ALNS Algorithm with LLM-Generated Operators';

  const slidesDir = 'D:/pythonProject/or_llm_agent/ppt_slides';

  console.log('Creating presentation...\n');

  // Slide 1: Title
  console.log('[1/15] Adding title slide...');
  await html2pptx(path.join(slidesDir, 'slide01_title.html'), pptx);

  // Slide 2: TOC
  console.log('[2/15] Adding table of contents...');
  await html2pptx(path.join(slidesDir, 'slide02_toc.html'), pptx);

  // Slide 3: Overview
  console.log('[3/15] Adding overview...');
  await html2pptx(path.join(slidesDir, 'slide03_overview.html'), pptx);

  // Slide 4: Innovations
  console.log('[4/15] Adding innovations...');
  await html2pptx(path.join(slidesDir, 'slide04_innovations.html'), pptx);

  // Slide 5: Architecture
  console.log('[5/15] Adding architecture...');
  await html2pptx(path.join(slidesDir, 'slide05_architecture.html'), pptx);

  // Slide 6: Operators
  console.log('[6/15] Adding operators...');
  await html2pptx(path.join(slidesDir, 'slide06_operators.html'), pptx);

  // Slide 7: Cost Model
  console.log('[7/15] Adding cost model...');
  await html2pptx(path.join(slidesDir, 'slide07_cost.html'), pptx);

  // Slide 8: Prompt Engineering
  console.log('[8/15] Adding prompt engineering...');
  await html2pptx(path.join(slidesDir, 'slide08_prompt.html'), pptx);

  // Slide 9: Development Journey
  console.log('[9/15] Adding development journey...');
  await html2pptx(path.join(slidesDir, 'slide09_development.html'), pptx);

  // Slide 10: Problem Classification
  console.log('[10/15] Adding problem classification...');
  await html2pptx(path.join(slidesDir, 'slide10_problems.html'), pptx);

  // Slide 11: Results
  console.log('[11/15] Adding results...');
  await html2pptx(path.join(slidesDir, 'slide11_results.html'), pptx);

  // Slide 12: Operator Performance Charts
  console.log('[12/15] Adding operator performance charts...');
  const { slide: slide12, placeholders } = await html2pptx(path.join(slidesDir, 'slide12_operators_chart.html'), pptx);

  // Add charts to placeholders
  if (placeholders.length >= 2) {
    // Destroy operators chart
    const destroyData = [
      { name: 'random_removal', labels: ['使用次数', '成功次数'], values: [632, 304] },
      { name: 'route_removal', labels: ['使用次数', '成功次数'], values: [60, 5] },
      { name: 'string_removal', labels: ['使用次数', '成功次数'], values: [317, 86] }
    ];

    slide12.addChart(pptx.charts.BAR, destroyData, {
      ...placeholders[0],
      barDir: 'col',
      showTitle: false,
      showLegend: true,
      legendPos: 'b',
      chartColors: ['B165FB', '40695B', '181B24'],
      catAxisLabelFontSize: 11,
      valAxisLabelFontSize: 11
    });

    // Repair operators chart
    const repairData = [
      { name: 'greedy_insert', labels: ['使用次数', '成功次数'], values: [237, 73] },
      { name: 'regret_insert', labels: ['使用次数', '成功次数'], values: [763, 322] }
    ];

    slide12.addChart(pptx.charts.BAR, repairData, {
      ...placeholders[1],
      barDir: 'col',
      showTitle: false,
      showLegend: true,
      legendPos: 'b',
      chartColors: ['B165FB', '40695B'],
      catAxisLabelFontSize: 11,
      valAxisLabelFontSize: 11
    });
  }

  // Slide 13: Conclusions
  console.log('[13/15] Adding conclusions...');
  await html2pptx(path.join(slidesDir, 'slide13_conclusions.html'), pptx);

  // Slide 14: Future Work
  console.log('[14/15] Adding future work...');
  await html2pptx(path.join(slidesDir, 'slide14_future.html'), pptx);

  // Slide 15: Thank You
  console.log('[15/15] Adding thank you slide...');
  await html2pptx(path.join(slidesDir, 'slide15_thankyou.html'), pptx);

  // Save presentation
  const outputPath = 'D:/pythonProject/or_llm_agent/ALNS_VRP_项目报告.pptx';
  console.log('\nSaving presentation...');
  await pptx.writeFile({ fileName: outputPath });
  
  console.log(`\n✅ Presentation created successfully!`);
  console.log(`📁 Location: ${outputPath}`);
  console.log(`📊 Total slides: 15`);
}

createPresentation().catch(err => {
  console.error('Error creating presentation:', err);
  process.exit(1);
});