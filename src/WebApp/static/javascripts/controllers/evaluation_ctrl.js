cbirCtrls.controller('EvaluationCtrl', ['$scope', '$http', function($scope, $http) {

  $scope.metrics = {
       precision : true,
       recal : false
     };
  $scope.algos = {
       convnet_oasis : true,
       bag_oasis : false
     };
  $scope.datasets = {
       mnist : true,
       paris : false
     };
  $scope.data = {}

  $scope.submit = function() {

        submit_metrics = false
        submit_datasets = false
        submit_algos = false
        metrics = []
        algos = []
        $.each( $scope.metrics, function( key, value ) {

          if (value == true) {
            submit_metrics = true
            metrics.push(key)
          }
        });

        $.each( $scope.datasets, function( key, value ) {

          if (value == true) {
            submit_datasets = true
            dataset = key
          }
        });

        $.each( $scope.algos, function( key, value ) {

          if (value == true) {
            submit_algos = true
            algos.push(key)
          }
        });

        if (submit_metrics && submit_datasets && submit_algos) {

          params = {
          'dataset': dataset,
          'algos': algos,
          'metrics': metrics,
        }

          $http.post('/evaluation', params)
              .then(function successCallback(response) {

              $.each(response.data, function(key_metric, arrays) {

                dataset = []

                for (i = 0; i < arrays[0].length; i++) {
                        elem = {
                          'x': i,
                        }
                        dataset.push(elem)
                }

                $.each(arrays, function(id, metric_array) {

                  real_key = 'val'.concat(id.toString())

                  $.each(metric_array, function(key, value) {

                    dataset[key][real_key] = value
                  })
                });

                $scope.data[key_metric.toString()] = dataset

                // console.log($('#'.concat(key_metric)))
                // $('#'.concat(key_metric)).html('<p>adad</p>')

              });

              // $scope.precision = {
              //         margin: {top: 5},
              //         series: [
              //           {
              //             axis: "y",
              //             dataset: "precision",
              //             key: "val1",
              //             label: "A line series",
              //             color: "hsla(88, 48%, 48%, 1)",
              //             type: ["line"],
              //             id: "mySeries1"
              //           },
              //           {
              //             axis: "y",
              //             dataset: "precision",
              //             key: "val0",
              //             label: "A another line series",
              //             color: "hsla(88, 48%, 48%, 1)",
              //             type: ["line"],
              //             id: "mySeries0"
              //           }
              //         ],
              //         axes: {x: {key: "x"}}
              //       };
          });



        }
      };
}]);
